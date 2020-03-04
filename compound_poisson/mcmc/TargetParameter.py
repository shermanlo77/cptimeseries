import numpy as np
from scipy.stats import norm

from .Target import Target
from .Target import get_parameter_mean_prior
from .Target import get_parameter_std_prior

class TargetParameter(Target):
    
    def __init__(self, time_series):
        super().__init__()
        self.time_series = time_series
        self.prior_mean = None
        self.prior_cov_chol = (get_parameter_std_prior()
            * np.ones(self.get_n_dim()))
        self.cp_parameter_before = None
        
        parameter_name_array = self.time_series.get_parameter_vector_name()
        self.prior_mean = get_parameter_mean_prior(parameter_name_array)
    
    def get_n_dim(self):
        return self.time_series.n_parameter
    
    def get_state(self):
        return self.time_series.get_parameter_vector()
    
    def update_state(self, state):
        self.time_series.set_parameter_vector(state)
        self.time_series.update_all_cp_parameters()
    
    def get_log_likelihood(self):
        return self.time_series.get_joint_log_likelihood()
    
    def get_log_target(self):
        ln_l = self.get_log_likelihood()
        ln_prior = self.get_log_prior(self.prior_cov_chol)
        return ln_l + ln_prior
    
    def get_log_prior(self, prior_cov_chol):
        return np.sum(
            norm.logpdf(self.get_state(), self.prior_mean, prior_cov_chol))
    
    def save_state(self):
        self.cp_parameter_before = self.time_series.copy_parameter()
    
    def revert_state(self):
        self.time_series.set_parameter(self.cp_parameter_before)
    
    def simulate_from_prior(self, rng):
        return rng.normal(self.prior_mean, self.prior_cov_chol)
