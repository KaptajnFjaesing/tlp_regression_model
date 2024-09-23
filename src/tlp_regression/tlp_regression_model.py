"""
Created on Mon Sep 16 07:36:48 2024

@author: Jonas Petersen
"""

import numpy as np
from pathlib import Path
import json
import pymc as pm
import arviz as az
from typing import (
    Dict,
    Tuple,
    Union
    )
import logging

from tlp_regression.config import (
    get_default_model_config,
    get_default_sampler_config,
    serialize_model_config,
)
from tlp_regression.model_components import (
    swish
)
from tlp_regression.utils import generate_hash_id 


class TlpRegressionModel:

    def __init__(
        self,
        model_config: dict | None = None,
        model_name: str = "TlpRegressionModel",
        model_version: str = None
        ):
        self.sampler_config = None
        self.model_config = (get_default_model_config() if model_config is None else model_config)  # parameters for priors, etc.
        self.model = None  # Set by build_model
        self.idata: az.InferenceData | None = None  # idata is generated during fitting
        self.posterior_predictive: az.InferenceData
        self.model_name = model_name
        self.model_version = model_version
        self.map_estimate = None
        self.logger = logging.getLogger("pymc")

    def build_model(self, X, y):
        with pm.Model() as self.model:
            (number_of_input_observations, number_of_features) = X.shape
            (number_of_output_observations, number_of_time_series_forecasts) = y.shape
            x = pm.Data("input", X, dims = ['number_of_input_observations', 'number_of_features'])
            y = pm.Data("target", y, dims = ['number_of_output_observations', 'number_of_time_series_forecasts'])
            precision_hidden_w1 = pm.Gamma('precision_hidden_w1', alpha = self.model_config["w_input_alpha_prior"], beta = self.model_config["w_input_beta_prior"])
            precision_hidden_b1 = pm.Gamma('precision_hidden_b1', alpha = self.model_config["b_input_alpha_prior"], beta = self.model_config["b_input_beta_prior"])
            W_hidden1 = pm.Normal('W_hidden1', mu=0, sigma=1/precision_hidden_w1, shape=(self.model_config["n_hidden_layer1"], number_of_features))
            b_hidden1 = pm.Normal('b_hidden1', mu=0, sigma=1/precision_hidden_b1, shape=(self.model_config["n_hidden_layer1"],))
            linear_layer1 = pm.math.dot(x, W_hidden1.T) + b_hidden1
            if self.model_config["activation"] == 'relu':
                hidden_layer1 = pm.math.maximum(linear_layer1, 0)
            elif self.model_config["activation"] == 'swish':
                hidden_layer1 = swish(linear_layer1)
            else:
                raise ValueError("Unsupported activation function")
            precision_hidden_w2 = pm.Gamma('precision_hidden_w2', alpha = self.model_config["w2_alpha_prior"], beta = self.model_config["w2_beta_prior"])
            precision_hidden_b2 = pm.Gamma('precision_hidden_b2', alpha = self.model_config["b2_alpha_prior"], beta = self.model_config["b2_beta_prior"])
            W_hidden2 = pm.Normal('W_hidden2', mu=0, sigma=1/precision_hidden_w2, shape=(self.model_config["n_hidden_layer2"], self.model_config["n_hidden_layer1"]))
            b_hidden2 = pm.Normal('b_hidden2', mu=0, sigma=1/precision_hidden_b2, shape=(self.model_config["n_hidden_layer2"],))
            linear_layer2 = pm.math.dot(hidden_layer1, W_hidden2.T) + b_hidden2
            if self.model_config["activation"] == 'relu':
                hidden_layer2 = pm.math.maximum(linear_layer2, 0)
            elif self.model_config["activation"] == 'swish':
                hidden_layer2 = swish(linear_layer2)
            else:
                raise ValueError("Unsupported activation function")
            precision_output_w = pm.Gamma('precision_output_w', alpha = self.model_config["w_output_alpha_prior"], beta = self.model_config["w_output_beta_prior"])
            precision_output_b = pm.Gamma('precision_output_b', alpha = self.model_config["b_output_alpha_prior"], beta = self.model_config["b_output_beta_prior"])
            W_output = pm.Normal('W_output', mu=0, sigma=1/precision_output_w, shape=(number_of_time_series_forecasts, self.model_config["n_hidden_layer2"]))
            b_output = pm.Normal('b_output', mu=0, sigma=1/precision_output_b, shape=(number_of_time_series_forecasts,))
            target_mean = pm.math.dot(hidden_layer2, W_output.T) + b_output
            precision_obs = pm.Gamma('precision_obs', alpha = self.model_config["precision_alpha_prior"], beta = self.model_config["precision_beta_prior"])
            pm.Normal('target_distribution', mu = target_mean, sigma = 1/precision_obs,observed = y, dims = ['number_of_input_observations', 'number_of_time_series_forecasts'])

    def fit(
        self,
        X,
        y,
        sampler_config: dict | None = None,
    ) -> az.InferenceData:
        self.build_model(X = X, y = y)
        self.sampler_config = (get_default_sampler_config() if sampler_config is None else sampler_config)
        if not self.sampler_config['verbose']:
            self.logger.setLevel(logging.CRITICAL)
            self.sampler_config['progressbar'] = False
        else:
            self.logger.setLevel(logging.INFO)
            self.sampler_config['progressbar'] = True
        
        with self.model:
            if self.sampler_config['sampler'] == "MAP":
                self.map_estimate = [pm.find_MAP(progressbar = self.sampler_config['progressbar'])]
            else:
                if self.sampler_config['sampler'] == "NUTS":
                    step=pm.NUTS()
                if self.sampler_config['sampler'] == "HMC":
                    step=pm.HamiltonianMC()
                if self.sampler_config['sampler'] == "metropolis":
                    step=pm.Metropolis()
                idata_temp = pm.sample(step = step, **{k: v for k, v in self.sampler_config.items() if (k != 'sampler' and k != 'verbose')})
                self.idata = self.set_idata_attrs(idata_temp)

    def set_idata_attrs(self, idata=None):
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs["id"] = self.id
        idata.attrs["model_name"] = self.model_name
        idata.attrs["model_version"] = self.model_version
        idata.attrs["sampler_config"] = serialize_model_config(self.sampler_config)
        idata.attrs["model_config"] = serialize_model_config(self._serializable_model_config)
        return idata

    def sample_posterior_predictive(
        self,
        x_test,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self.model:
            pm.set_data({'input': x_test})
            if self.sampler_config['sampler'] == "MAP":
                if not self.sampler_config['verbose']:
                    self.posterior_predictive = pm.sample_posterior_predictive(self.map_estimate, predictions=True, progressbar = False, **kwargs)
                else:
                    self.posterior_predictive = pm.sample_posterior_predictive(self.map_estimate, predictions=True, progressbar = True, **kwargs)
            else:
                if not self.sampler_config['verbose']:
                    self.posterior_predictive = pm.sample_posterior_predictive(self.idata, predictions=True, progressbar = False, **kwargs)
                else:
                    self.posterior_predictive = pm.sample_posterior_predictive(self.idata, predictions=True, progressbar = True, **kwargs)
        return self.posterior_predictive.predictions
    
    def get_posterior_predictive(self) -> az.InferenceData:
        return self.posterior_predictive

    @property
    def id(self) -> str:
        return generate_hash_id(self.model_config, self.model_version, self.model_name)

    @property
    def output_var(self):
        return "target_distribution"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        return self.model_config
    
    def get_model(self):
        return self.model
    
    def get_idata(self):
        return self.idata
    
    def save(self, fname: str) -> None:
        if self.idata is not None and "posterior" in self.idata:
            if self.method == "MAP":
                raise RuntimeError("The MAP method cannot be saved.")
            file = Path(str(fname))
            self.idata.to_netcdf(str(file))
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
                

    def load(self, fname: str):
        filepath = Path(str(fname))
        self.idata = az.from_netcdf(filepath)
        self.model_config = json.loads(self.idata.attrs["model_config"])
        self.sampler_config = json.loads(self.idata.attrs["sampler_config"])
        
        self.build_model(
            X = self.idata.constant_data.input,
            y = self.idata.constant_data.target
            )

