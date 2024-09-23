"""
Created on Tue Sep 10 19:28:04 2024

@author: Jonas Petersen
"""

import json
from typing import Dict

def get_default_model_config() -> Dict:
    base_alpha = 2
    base_beta = 0.1
    
    model_config: Dict = {
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
        "n_hidden_layer1": 20,
        "n_hidden_layer2": 20
    }
    
    return model_config


def get_default_sampler_config() -> Dict:
    sampler_config: Dict = {
        "draws": 1000,
        "tune": 200,
        "chains": 1,
        "cores": 1,
        "sampler": "NUTS",
        "nuts_sampler": "pymc",
        "verbose": True
    }
    return sampler_config


def serialize_model_config(config: Dict) -> str:
    return json.dumps(config)


def deserialize_model_config(config_str: str) -> Dict:
    return json.loads(config_str)
