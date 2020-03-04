import numpy as np

from .Target import Target
from .Target import get_arma_index
from .Target import get_precision_prior

class TargetDownscalePrecision(Target):
    
    def __init__(self, target_downscale_parameter):
        super().__init__()
        self.target_parameter = target_downscale_parameter
        #reg then arma
        self.prior = get_precision_prior()
        self.precision = []
        for prior in self.prior:
            self.precision.append(prior.mean())
        self.precision = np.asarray(self.precision)
        self.precision_before = None
    
    def get_n_dim(self):
        return len(self.precision)
    
    def get_state(self):
        return self.precision
    
    def update_state(self, state):
        self.precision = state
    
    def save_state(self):
        self.precision_before = self.precision.copy()
    
    def revert_state(self):
        self.precision = self.precision_before
    
    def simulate_from_prior(self, rng):
        prior_simulate = []
        for prior in self.prior:
            prior.random_state = rng
            prior_simulate.append(prior.rvs())
        return np.asarray(prior_simulate)
