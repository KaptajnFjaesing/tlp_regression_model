"""
Created on Tue Sep 10 19:24:46 2024

@author: Jonas Petersen
"""

import numpy as np
import pandas as pd
import hashlib
import json
from typing import (
    Dict, 
    Union
    )

def generate_hash_id(
    model_config: Dict[str, Union[int, float, Dict]],
    version: str,
    model_type: str
) -> str:
    """
    Generates a unique hash ID for a model based on its configuration, version, and type.

    Parameters
    ----------
    model_config : dict
        The configuration dictionary of the model.
    version : str
        The version of the model.
    model_type : str
        The type of the model.

    Returns
    -------
    str
        A unique hash string representing the model.
    """
    # Serialize the model configuration to a JSON string
    config_str = json.dumps(model_config, sort_keys=True)
    # Create a string to hash, combining the model type, version, and configuration
    hash_input = f"{model_type}-{version}-{config_str}"
    # Generate a unique hash ID
    hash_id = hashlib.md5(hash_input.encode()).hexdigest()
    return hash_id

def compute_residuals(model_forecasts, test_data, min_forecast_horizon):
    """
    Compute the forecast error metrics for multiple models.

    Parameters:
    - model_forecasts: List of forecasted values from different models. Each element is a list of forecasts for each column.
    - test_data: DataFrame containing the actual values for the forecast period.
    - min_forecast_horizon: The minimum forecast horizon to consider for each model.

    Returns:
    - List of DataFrames, each containing the forecast errors for a model.
    """
    metrics_list = []

    # Iterate over each model's forecasts
    for forecasts in model_forecasts:
        # Initialize an empty DataFrame to store forecast errors for the current model
        errors_df = pd.DataFrame(columns=test_data.columns)
        
        # Calculate forecast errors for each column
        for i, column in enumerate(test_data.columns):
            # Get the forecast horizon for the current column
            forecast_horizon = len(forecasts[i].values)
            
            # Calculate forecast errors: actual values minus forecasted values
            errors_df[column] = test_data[column].values[-forecast_horizon:] - forecasts[i].values
        
        # Keep only the first `min_forecast_horizon` rows and reset index
        truncated_errors_df = errors_df.iloc[:min_forecast_horizon].reset_index(drop=True)
        
        # Append the DataFrame of errors for the current model to the list
        metrics_list.append(truncated_errors_df)
    stacked = pd.concat(metrics_list, axis=0)
    
    return stacked

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