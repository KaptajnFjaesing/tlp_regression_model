"""
Created on Mon Sep 16 08:37:30 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
from examples.load_data import normalized_weekly_store_category_household_sales

df = normalized_weekly_store_category_household_sales()

# %%
def create_lagged_features(df, time_series_columns, context_length, forecast_horizon, seasonality_period):
    """Create lagged features for a given column in the DataFrame, with future targets."""
    
    X = pd.DataFrame()
    y = pd.DataFrame()
    
    data_min = df.min()
    data_max = df.max()
    
    for column_name in time_series_columns:
        for i in range(context_length, -1, -1):
            X[f'lag_{column_name}_{i}'] = (df[column_name].shift(i)-data_min[column_name])/(data_max[column_name]-data_min[column_name])
        
        for i in range(1, forecast_horizon + 1):
            y[f'target_{column_name}_{i}'] = (df[column_name].shift(-i)-data_min[column_name])/(data_max[column_name]-data_min[column_name])
    X['time_sine'] = np.sin(2*np.pi*X.index/seasonality_period)
    X.dropna(inplace=True)  # Remove rows with NaN values due to shifting
    y.dropna(inplace=True)  # Remove rows with NaN values due to shifting
    return X.iloc[:-forecast_horizon], y.iloc[context_length:]

# Parameters
context_length = 35  # Number of lagged time steps to use as features
forecast_horizon = 26  # Number of future time steps to predict
seasonality_period = 52

time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

df_time_series = df[time_series_columns]

training_data = df_time_series.iloc[:-forecast_horizon]
test_data = df_time_series.iloc[-forecast_horizon:]

(x_train,y_train) = create_lagged_features(
    df = training_data,
    time_series_columns = unnormalized_column_group,
    context_length = context_length,
    forecast_horizon = forecast_horizon,
    seasonality_period = seasonality_period
   )


(x_total,y_total) = create_lagged_features(
    df = df_time_series,
    time_series_columns = unnormalized_column_group,
    context_length = context_length,
    forecast_horizon = forecast_horizon,
    seasonality_period = seasonality_period
   )

test_input_index = training_data.iloc[-1:].index[0]

x_test = x_total[x_total.index == test_input_index]
y_test = y_total[y_total.index == test_input_index]


# %% Define model
from src.tlp_regression_model import TlpRegressionModel


model_name = "TlpRegressionModel"
version = "v0.1"

# Sorcerer
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
    "n_hidden_layer1": 20,
    "n_hidden_layer2": 20
}

if sampler_config['sampler'] == "MAP":
    model_config['precision_alpha_prior'] = 100
    model_config['precision_beta_prior'] = 0.001

prognosticator = TlpRegressionModel(
    model_config = model_config,
    sampler_config = sampler_config,
    model_name = model_name,
    version = version
    )


# %% Fit model

prognosticator.fit(
    X = x_train,
    y = y_train
    )

"""

if method != "MAP":
    fname = "examples/models/sorcer_model_v02.nc"
    prognosticator.save(fname)

"""

#%% Produce forecast

model_preds = prognosticator.sample_posterior_predictive(x_test = x_test)


#%%

raw_model_preds = model_preds["target_distribution"].mean(("chain", "draw"))

raw_model_preds_pandas = pd.DataFrame(data = raw_model_preds, columns = y_train.columns)


list_of_forecasts = [pd.Series(raw_model_preds_pandas.filter(like=column_name).values[0].T) for column_name in unnormalized_column_group]
list_of_targets = [pd.Series(y_test.filter(like=column_name).values[0].T) for column_name in unnormalized_column_group]

data = pd.DataFrame()
data_min = df.min()
data_max = df.max()
for i in range(len(unnormalized_column_group)):
    data[f'normalized{unnormalized_column_group[i]}'] = (df[unnormalized_column_group[i]]-data_min[unnormalized_column_group[i]])/(data_max[unnormalized_column_group[i]]-data_min[unnormalized_column_group[i]])



hdi_values = az.hdi(model_preds)["target_distribution"].transpose("hdi", ...)

n_cols = 2  # We want 2 columns
n_rows = int(np.ceil((len(time_series_columns)-1) / n_cols))  # Number of rows needed
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
axs = axs.flatten()
# Loop through each column to plot
for i in range(len(time_series_columns)-1):
    ax = axs[i]  # Get the correct subplot
    
    test_train_split = y_total[y_total.columns[i]].index.max()
    
    ax.plot(data[data.columns[i]].iloc[:test_train_split], color = 'tab:red',  label='Training Data')
    ax.plot(data[data.columns[i]].iloc[test_train_split:], color = 'black',  label='Test Data')
    ax.plot(np.arange(
        y_total[y_total.columns[i]].index.max()+1,
        y_total[y_total.columns[i]].index.max()+len(list_of_targets[0])+1),
        list_of_forecasts[i].values,
        color = 'tab:blue',
        label='Model')
    #ax.fill_between(
    #    np.arange(test_input_index,test_input_index+forecast_horizon),
    #    hdi_values[0].values[:,i],  # lower bound of the HDI
    #    hdi_values[1].values[:,i],  # upper bound of the HDI
    #    color= 'blue',   # color of the shaded region
    #    alpha=0.4,      # transparency level of the shaded region
    #)
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure



#%% Plot forecast along with test data


hdi_values = az.hdi(model_preds)["target_distribution"].transpose("hdi", ...)

# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil((len(time_series_columns)-1) / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows   
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(len(time_series_columns)-1):
    ax = axs[i]  # Get the correct subplot
    
    ax.plot(np.arange(0,test_input_index+1), training_data[unnormalized_column_group[i]], color = 'tab:red',  label='Training Data')
    ax.plot(np.arange(test_input_index,test_input_index+forecast_horizon), test_data[unnormalized_column_group[i]], color = 'black',  label='Test Data')
    ax.plot(np.arange(test_input_index,test_input_index+forecast_horizon), (model_preds["target_distribution"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
    #ax.fill_between(
    #    np.arange(test_input_index,test_input_index+forecast_horizon),
    #    hdi_values[0].values[:,i],  # lower bound of the HDI
    #    hdi_values[1].values[:,i],  # upper bound of the HDI
    #    color= 'blue',   # color of the shaded region
    #    alpha=0.4,      # transparency level of the shaded region
    #)
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure



# %%
plt.savefig('./examples/figures/forecast.png')
