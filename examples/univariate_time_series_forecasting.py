"""
Created on Wed Sep 18 12:23:09 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from examples.load_data import load_m5_weekly_store_category_sales_data

_,df,_ = load_m5_weekly_store_category_sales_data()

# %% Define model

from src.TlpRegressionModel.utils import create_lagged_features
from src.TlpRegressionModel.tlp_regression_model import TlpRegressionModel


model_name = "TlpRegressionModel"
version = "v0.1"

sampler_config = {
    "draws": 200,
    "tune": 100,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "nuts_sampler": "numpyro"
}

model_config = {
    "w_input_alpha_prior": 2,
    "b_input_alpha_prior": 2,
    "w2_alpha_prior": 2,
    "b2_alpha_prior": 2,
    "w_output_alpha_prior": 2,
    "b_output_alpha_prior": 2,
    "precision_alpha_prior": 2,
    "w_input_beta_prior": 0.1,
    "b_input_beta_prior": 0.1,
    "w2_beta_prior": 0.1,
    "b2_beta_prior": 0.1,
    "w_output_beta_prior": 0.1,
    "b_output_beta_prior": 0.1,
    "precision_beta_prior": 0.1,
    "activation": "swish",
    "n_hidden_layer1": 5,
    "n_hidden_layer2": 5
}


prognosticator = TlpRegressionModel(
    model_config = model_config,
    sampler_config = sampler_config,
    model_name = model_name,
    version = version
    )


time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

df_time_series = df[time_series_columns]

seasonality_period = 52
context_length = 35  # Number of lagged time steps to use as features
min_forecast_horizon = 26
max_forecast_horizon = 52

model_forecasts = []
for forecast_horizon in range(min_forecast_horizon,max_forecast_horizon+1):
    print("Forecast horizon", forecast_horizon)
    training_data = df_time_series.iloc[:-forecast_horizon]
    test_data = df_time_series.iloc[-forecast_horizon:]
    model_denormalized = []
    for time_series in unnormalized_column_group:
        (x_train,y_train) = create_lagged_features(
            df = training_data,
            time_series_columns = [time_series],
            context_length = context_length,
            forecast_horizon = forecast_horizon,
            seasonality_period = seasonality_period
           )
        
        (x_total,y_total) = create_lagged_features(
            df = df_time_series,
            time_series_columns = [time_series],
            context_length = context_length,
            forecast_horizon = forecast_horizon,
            seasonality_period = seasonality_period
           )
        prognosticator.fit(
            X = x_train,
            y = y_train
            )
        model_preds = prognosticator.sample_posterior_predictive(x_test = x_total[x_total.index == training_data.iloc[-1:].index[0]])
        raw_model_preds_pandas = pd.DataFrame(data = model_preds["target_distribution"].mean(("chain", "draw")), columns = y_train.columns)        
        model_denormalized.append(pd.Series(model_preds["target_distribution"].mean(("chain", "draw")).values[0])*((training_data[time_series].max()-training_data[time_series].min()))+training_data[time_series].min())
    model_forecasts.append(model_denormalized)


#%% Compute MASEs
from src.utils import compute_residuals

abs_mean_gradient_training_data = pd.read_pickle('./data/abs_mean_gradient_training_data.pkl')

stacked_sorcerer = compute_residuals(
         model_forecasts = model_forecasts,
         test_data = test_data[unnormalized_column_group],
         min_forecast_horizon = min_forecast_horizon
         )

average_abs_residual_sorcerer = stacked_sorcerer.abs().groupby(stacked_sorcerer.index).mean() # averaged over rolls
average_abs_residual_sorcerer.columns = unnormalized_column_group
MASE_sorcerer = average_abs_residual_sorcerer/abs_mean_gradient_training_data

print("avg MASE_sorcerer: ", MASE_sorcerer.mean(axis = 1).mean())

# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil((len(time_series_columns)-1) / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows   
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(len(unnormalized_column_group)):
    ax = axs[i]  # Get the correct subplot
    ax.plot(MASE_sorcerer[unnormalized_column_group[i]], color = 'tab:blue',  label='MASE Sorcerer')
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.set_title(unnormalized_column_group[i])
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure


#%%
stacked_sorcerer.to_pickle(r'C:\Users\roman\Documents\git\TimeSeriesForecastingReview\data\results\stacked_forecasts_tlp_univariate.pkl')
