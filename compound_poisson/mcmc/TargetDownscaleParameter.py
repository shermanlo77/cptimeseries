import os
import sys

import numpy as np

import compound_poisson as cp
from .Target import Target
from .Target import get_arma_index
from .Target import get_parameter_mean_prior
from .Target import get_parameter_std_prior

class TargetDownscaleParameter(Target):
    
    def __init__(self, downscale):
        super().__init__()
        self.downscale = downscale
        self.n_parameter = downscale.n_parameter
        self.n_total_parameter = downscale.n_total_parameter
        self.area_unmask = downscale.area_unmask
        self.prior_mean = None
        self.prior_cov_chol = None
        self.prior_scale_parameter = get_parameter_std_prior()
        self.prior_scale_arma = get_parameter_std_prior()
        self.parameter_before = None
        self.arma_index = None
        
        parameter_name_array = self.downscale.get_parameter_vector_name()
        self.prior_mean = get_parameter_mean_prior(parameter_name_array)
        self.prior_cov_chol = np.identity(self.area_unmask)
        self.arma_index = get_arma_index(parameter_name_array)
    
    def get_n_dim(self):
        return self.n_total_parameter
    
    def get_state(self):
        return self.downscale.get_parameter_vector()
    
    def update_state(self, state):
        self.downscale.set_parameter_vector(state)
        self.downscale.update_all_cp_parameters()
    
    def simulate_from_prior(self, rng):
        parameter_vector = self.prior_mean.copy()
        for i in range(self.n_parameter):
            parameter_i = np.asarray(rng.normal(size=self.area_unmask))
            chol = self.prior_cov_chol.copy()
            if self.arma_index[i*self.area_unmask]:
                chol *= self.prior_scale_arma
            else:
                chol *= self.prior_scale_parameter
            parameter_i = np.matmul(chol, parameter_i)
            parameter_vector[i*self.area_unmask : (i+1)*self.area_unmask] += (
                parameter_i)
        return parameter_vector.flatten()
