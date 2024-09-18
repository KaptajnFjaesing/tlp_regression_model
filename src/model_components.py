"""
Created on Tue Sep 10 19:26:17 2024

@author: Jonas Petersen
"""

import numpy as np

def swish(x):
    return x / (1.0 + np.exp(-x))
