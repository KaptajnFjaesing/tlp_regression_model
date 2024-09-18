"""
Created on Tue Sep 10 19:28:04 2024

@author: Jonas Petersen
"""

import json
from typing import Dict

def get_default_model_config() -> Dict:
    """
    Returns the default model configuration as a dictionary.
    
    The configuration includes parameters for priors, the number of components, 
    forecast horizon, and other hyperparameters used in the model.
    
    Returns:
    - A dictionary containing the default model configuration.
    """
    model_config: Dict = {
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
        "activation": "relu",
        "n_hidden_layer1": 20,
        "n_hidden_layer2": 20
    }
    return model_config


def get_default_sampler_config() -> Dict:
    """
    Returns the default sampler configuration as a dictionary.
    
    The configuration includes parameters for MCMC sampling such as the number of draws, 
    tuning steps, chains, cores, and target acceptance rate.
    
    Returns:
    - A dictionary containing the default sampler configuration.
    """
    sampler_config: Dict = {
        "draws": 1000,
        "tune": 200,
        "chains": 1,
        "cores": 1,
        "sampler": "NUTS",
        "nuts_sampler": "pymc"
    }
    return sampler_config


def serialize_model_config(config: Dict) -> str:
    """
    Serializes the model configuration dictionary to a JSON string.
    
    Parameters:
    - config: A dictionary containing the model configuration.
    
    Returns:
    - A JSON string representation of the model configuration.
    """
    return json.dumps(config)


def deserialize_model_config(config_str: str) -> Dict:
    """
    Deserializes a JSON string to a model configuration dictionary.
    
    Parameters:
    - config_str: A JSON string representation of the model configuration.
    
    Returns:
    - A dictionary containing the model configuration.
    """
    return json.loads(config_str)
