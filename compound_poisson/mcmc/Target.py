import numpy as np
import scipy.stats as stats

import compound_poisson as cp

class Target:
    
    def __init__(self):
        self.prior_mean = None #used by slice sampling
        self.prior_cov_chol = None #used by slice sampling
    
    def get_n_dim(self):
        raise NotImplementedError
    
    def get_state(self):
        raise NotImplementedError
    
    def update_state(self, state):
        raise NotImplementedError
    
    def get_log_likelihood(self):
        raise NotImplementedError
    
    def get_log_target(self):
        raise NotImplementedError
    
    def save_state(self):
        raise NotImplementedError
    
    def revert_state(self):
        raise NotImplementedError
    
    def simulate_from_prior(self, rng):
        raise NotImplementedError

def get_parameter_mean_prior(parameter_name_array):
    prior_mean = np.zeros(len(parameter_name_array))
    for i, parameter_name in enumerate(parameter_name_array):
        if parameter_name.endswith("const"):
            if cp.PoissonRate.__name__ in parameter_name:
                prior_mean[i] = -0.46
            elif cp.GammaMean.__name__ in parameter_name:
                prior_mean[i] = 1.44
            elif cp.GammaDispersion.__name__ in parameter_name:
                prior_mean[i] = -0.45
    return prior_mean

def get_parameter_std_prior():
    return 0.5

def get_precision_prior():
    prior = [
        stats.gamma(a=2.8, scale=2.3),
        stats.gamma(a=1.3, loc=16, scale=65),
    ]
    return prior

def get_arma_index(parameter_name_array):
    arma_index = []
    for parameter_name in parameter_name_array:
        if "_AR" in parameter_name:
            arma_index.append(True)
        elif "_MA" in parameter_name:
            arma_index.append(True)
        else:
            arma_index.append(False)
    return arma_index
    
