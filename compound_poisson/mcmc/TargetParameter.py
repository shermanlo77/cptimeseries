import numpy as np
from scipy.stats import norm

from .Target import Target

class TargetParameter(Target):
    
    def __init__(self, time_series):
        super().__init__()
        self.time_series = time_series
        self.prior_mean = np.zeros(self.get_n_dim())
        self.prior_cov_chol = np.sqrt(0.25 * np.ones(self.get_n_dim()))
        self.cp_parameter_before = None
        
        parameter_name_array = self.time_series.get_parameter_vector_name()
        for i, parameter_name in enumerate(parameter_name_array):
            if parameter_name.endswith("const"):
                if parameter_name.startswith(
                    self.time_series.poisson_rate.__class__.__name__):
                    self.prior_mean[i] = -0.46
                elif parameter_name.startswith(
                    self.time_series.gamma_mean.__class__.__name__):
                    self.prior_mean[i] = 1.44
                elif parameter_name.startswith(
                    self.time_series.gamma_dispersion.__class__.__name__):
                    self.prior_mean[i] = -0.45
    
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
