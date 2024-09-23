"""
Created on Mon Sep 23 12:49:54 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from examples.load_data import load_m5_weekly_store_category_sales_data

_,df,_ = load_m5_weekly_store_category_sales_data()

# %% Define model

from tlp_regression.tlp_regression_model import TlpRegressionModel

model_name = "TlpRegressionModel"
version = "v0.2"

sampler_config = {
    "draws": 200,
    "tune": 100,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "nuts_sampler": "numpyro"
}

base_alpha = 2
base_beta = 0.1

model_config = {
    "w_input_alpha_prior": base_alpha,
    "b_input_alpha_prior": base_alpha,
    "w2_alpha_prior": base_alpha,
    "b2_alpha_prior": base_alpha,
    "w_output_alpha_prior": base_alpha,
    "b_output_alpha_prior": base_alpha,
    "precision_alpha_prior": 10,
    "w_input_beta_prior": base_beta,
    "b_input_beta_prior": base_beta,
    "w2_beta_prior": base_beta,
    "b2_beta_prior": base_beta,
    "w_output_beta_prior": base_beta,
    "b_output_beta_prior": base_beta,
    "precision_beta_prior": 0.1,
    "activation": "swish",
    "n_hidden_layer1": 100,
    "n_hidden_layer2": 100
}


tlp_regression_model = TlpRegressionModel(
    model_config = model_config,
    sampler_config = sampler_config,
    model_name = model_name,
    version = version
    )


# %%

number_of_weeks_in_a_year = 52.1429
number_of_months = 12
number_of_quaters = 4


time_series_columns = [x for x in df.columns if not 'date' in x]
df_time_series = df.copy(deep = True)

df_time_series['week'] = np.sin(2*np.pi*df_time_series['date'].dt.strftime('%U').astype(int)/number_of_weeks_in_a_year)
df_time_series["month"] = np.sin(2*np.pi*df_time_series["date"].dt.month/number_of_months)
df_time_series["quarter"] = np.sin(2*np.pi*df_time_series["date"].dt.quarter/number_of_quaters)


fh = 30

training_data = df_time_series.iloc[:-fh]
test_data = df_time_series.iloc[-fh:]

x_training_min = training_data["date"].dt.year.min()
x_training_max = training_data["date"].dt.year.max()
training_data["year"] = (training_data["date"].dt.year-x_training_min)/(x_training_max-x_training_min)
test_data["year"] = (test_data["date"].dt.year-x_training_min)/(x_training_max-x_training_min)

y_training_min = training_data[time_series_columns].min()
y_training_max = training_data[time_series_columns].max()
x_train = training_data[["week", "month", "quarter", "year"]].values
y_train = ((training_data[time_series_columns]-y_training_min)/(y_training_max-y_training_min)).values

x_test = test_data[["week", "month", "quarter", "year"]].values
y_test = ((test_data[time_series_columns]-y_training_min)/(y_training_max-y_training_min)).values

tlp_regression_model.fit(
    X = x_train,
    y = y_train
    )


#%%

model_preds = tlp_regression_model.sample_posterior_predictive(x_test = x_test)


ko = [pd.Series((model_preds["target_distribution"].mean(("chain", "draw")).T)[i].values*(y_training_max[y_training_max.index[i]]-y_training_min[y_training_min.index[i]])+y_training_min[y_training_min.index[i]]) for i in range(len(time_series_columns))]

# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil((len(time_series_columns)) / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows   
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(len(time_series_columns)):
    ax = axs[i]  # Get the correct subplot
    ax.plot(df[time_series_columns[i]], color = 'tab:blue',  label='Training Data')
    ax.plot(range(len(df)-fh+1,len(df)+1), ko[i], color = 'black',  label='Test Data')
    #ax.plot(preds_out_of_sample, (model_preds["target_distribution"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
    ax.set_title(time_series_columns[i])
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure
