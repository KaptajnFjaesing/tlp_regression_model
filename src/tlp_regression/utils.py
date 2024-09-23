"""
Created on Tue Sep 10 19:24:46 2024

@author: Jonas Petersen
"""

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