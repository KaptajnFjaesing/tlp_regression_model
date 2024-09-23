"""
Created on Mon Sep 23 11:29:04 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from examples.load_data import load_m5_weekly_store_category_sales_data

_,df,_ = load_m5_weekly_store_category_sales_data()

# %% Define model

from src.tlp_regression_model.model_components import swish

import pymc as pm

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

base_alpha = 10
base_beta = 0.1

model_config = {
    "w_input_alpha_prior": base_alpha,
    "b_input_alpha_prior": base_alpha,
    "w2_alpha_prior": base_alpha,
    "b2_alpha_prior": base_alpha,
    "w_output_alpha_prior": base_alpha,
    "b_output_alpha_prior": base_alpha,
    "precision_alpha_prior": 100,
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



time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]

df_time_series = df[time_series_columns]

fh = 30
time_series = time_series_columns[1]


training_data = df_time_series.iloc[:-fh]
test_data = df_time_series.iloc[-fh:]

simulated_number_of_forecasts = 50
number_of_weeks_in_a_year = 52.1429
number_of_months = 12
number_of_quaters = 4

time_series_columns = [x for x in df.columns if not 'date' in x]
df_time_series = df.copy(deep = True)

df_time_series['week'] = np.sin(2*np.pi*df_time_series['date'].dt.strftime('%U').astype(int)/number_of_weeks_in_a_year)
df_time_series["month"] = np.sin(2*np.pi*df_time_series["date"].dt.month/number_of_months)
df_time_series["quarter"] = np.sin(2*np.pi*df_time_series["date"].dt.quarter/number_of_quaters)

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

X1 = x_train
y1 = y_train


#%%
with pm.Model() as model:
    (number_of_input_observations, number_of_features) = X1.shape
    (number_of_output_observations, number_of_time_series_forecasts) = y1.shape

    x = pm.Data("input", X1, dims = ['number_of_input_observations', 'number_of_features'])
    y = pm.Data("target", y1, dims = ['number_of_output_observations', 'number_of_time_series_forecasts'])
    # Priors for the weights and biases of the hidden layer 1
    precision_hidden_w1 = pm.Gamma('precision_hidden_w1', alpha = model_config["w_input_alpha_prior"], beta = model_config["w_input_beta_prior"])
    precision_hidden_b1 = pm.Gamma('precision_hidden_b1', alpha = model_config["b_input_alpha_prior"], beta = model_config["b_input_beta_prior"])
    
    # Hidden layer 1 weights and biases
    W_hidden1 = pm.Normal(
        'W_hidden1',
        mu=0,
        sigma=1/precision_hidden_w1,
        shape=(model_config["n_hidden_layer1"], number_of_features)
        )
    b_hidden1 = pm.Normal(
        'b_hidden1',
        mu=0,
        sigma=1/precision_hidden_b1,
        shape=(model_config["n_hidden_layer1"],)
        )
    
    # Compute the hidden layer 1 outputs
    linear_layer1 = pm.math.dot(x, W_hidden1.T) + b_hidden1
    
    if model_config["activation"] == 'relu':
        hidden_layer1 = pm.math.maximum(linear_layer1, 0)
    elif model_config["activation"] == 'swish':
        hidden_layer1 = swish(linear_layer1)
        #hidden_layer1 = linear_layer1 * pm.math.sigmoid(linear_layer1)
    else:
        raise ValueError("Unsupported activation function")

    # Priors for the weights and biases of the hidden layer 2
    precision_hidden_w2 = pm.Gamma('precision_hidden_w2', alpha = model_config["w2_alpha_prior"], beta = model_config["w2_beta_prior"])
    precision_hidden_b2 = pm.Gamma('precision_hidden_b2', alpha = model_config["b2_alpha_prior"], beta = model_config["b2_beta_prior"])
    
    # Hidden layer 2 weights and biases
    W_hidden2 = pm.Normal(
        'W_hidden2',
        mu=0,
        sigma=1/precision_hidden_w2,
        shape=(model_config["n_hidden_layer2"], model_config["n_hidden_layer1"])
        )
    b_hidden2 = pm.Normal(
        'b_hidden2',
        mu=0,
        sigma=1/precision_hidden_b2,
        shape=(model_config["n_hidden_layer2"],)
        )
    
    # Compute the hidden layer 2 outputs
    linear_layer2 = pm.math.dot(hidden_layer1, W_hidden2.T) + b_hidden2

    if model_config["activation"] == 'relu':
        hidden_layer2 = pm.math.maximum(linear_layer2, 0)
    elif model_config["activation"] == 'swish':
        hidden_layer2 = swish(linear_layer2)
        #hidden_layer2 = linear_layer1 * pm.math.sigmoid(linear_layer2)
    else:
        raise ValueError("Unsupported activation function")
    
    # Priors for the weights and biases of the output layer
    precision_output_w = pm.Gamma('precision_output_w', alpha = model_config["w_output_alpha_prior"], beta = model_config["w_output_beta_prior"])
    precision_output_b = pm.Gamma('precision_output_b', alpha = model_config["b_output_alpha_prior"], beta = model_config["b_output_beta_prior"])
    
    # Output layer weights and biases
    W_output = pm.Normal('W_output', mu=0, sigma=1/precision_output_w, shape=(number_of_time_series_forecasts, model_config["n_hidden_layer2"]))
    b_output = pm.Normal('b_output', mu=0, sigma=1/precision_output_b, shape=(number_of_time_series_forecasts,))

    # Compute the output (regression prediction)
    target_mean = pm.math.dot(hidden_layer2, W_output.T) + b_output
    
    # Likelihood (using Normal distribution for regression)
    precision_obs = pm.Gamma('precision_obs', alpha = model_config["precision_alpha_prior"], beta = model_config["precision_beta_prior"])
    pm.Normal(
        'target_distribution',
        mu = target_mean,
        sigma = 1/precision_obs,
        observed = y,
        dims = ['number_of_input_observations', 'number_of_time_series_forecasts']
        )

with model:
    idata_temp = pm.sample(step = pm.NUTS(), **{k: v for k, v in sampler_config.items() if k != 'sampler'})


with model:
    pm.set_data({'input': x_test})
    posterior_predictive = pm.sample_posterior_predictive(idata_temp, predictions=True)



# %%
ko = [pd.Series((posterior_predictive.predictions["target_distribution"].mean(("chain", "draw")).T)[i].values*(y_training_max[y_training_max.index[i]]-y_training_min[y_training_min.index[i]])+y_training_min[y_training_min.index[i]]) for i in range(len(time_series_columns))]

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

