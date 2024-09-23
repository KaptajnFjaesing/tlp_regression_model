"""
Created on Mon Sep 23 10:04:11 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from examples.load_data import load_m5_weekly_store_category_sales_data

_,df,_ = load_m5_weekly_store_category_sales_data()

# %% Define model

from src.tlp_regression_model.utils import create_lagged_features
from src.tlp_regression_model.model_components import swish

import pymc as pm

model_name = "TlpRegressionModel"
version = "v0.1"

sampler_config = {
    "draws": 500,
    "tune": 200,
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
    "precision_alpha_prior": 1000,
    "w_input_beta_prior": 0.1,
    "b_input_beta_prior": 0.1,
    "w2_beta_prior": 0.1,
    "b2_beta_prior": 0.1,
    "w_output_beta_prior": 0.1,
    "b_output_beta_prior": 0.1,
    "precision_beta_prior": 0.1,
    "activation": "swish",
    "n_hidden_layer1": 100,
    "n_hidden_layer2": 100
}


context_length = 30

time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

df_time_series = df[time_series_columns]

fh = 30
forecast_horizon = 30
seasonality_period = 52

time_series = time_series_columns[1]


training_data = df_time_series.iloc[:-fh]
test_data = df_time_series.iloc[-fh:]

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

X1 = x_train.values
y1 = y_train.values

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

#%%
with model:
    idata_temp = pm.sample(step = pm.NUTS(), **{k: v for k, v in sampler_config.items() if k != 'sampler'})


# %%
test_index = training_data.iloc[-1:].index[0]

chosen_index = 213
X_test = x_total[x_total.index == chosen_index].values
y_test = x_total[x_total.index == chosen_index].values

with model:
    pm.set_data({'input': X_test})
    posterior_predictive = pm.sample_posterior_predictive(idata_temp, predictions=True)

denormalized_result = pd.Series(posterior_predictive.predictions["target_distribution"].mean(("chain", "draw")).values[0])*((training_data[time_series].max()-training_data[time_series].min()))+training_data[time_series].min()

plt.figure()
plt.plot(df[time_series])
plt.plot(np.arange(chosen_index+1,chosen_index+1+forecast_horizon),denormalized_result)

print(posterior_predictive.predictions["target_distribution"].mean(("chain", "draw")).values)
print(y_test)

