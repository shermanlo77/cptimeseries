import math

import numpy as np
from numpy.linalg import cholesky
import scipy.stats as stats

from .Target import Target

class TargetDownscaleGp(Target):
    
    def __init__(self, downscale):
        self.downscale = downscale
        self.prior_precision = get_gp_precision_prior()
        self.precision = self.prior_precision.mean()
        self.area_unmask = downscale.area_unmask
        self.square_error = np.zeros((self.area_unmask, self.area_unmask))
        
        unmask = np.logical_not(downscale.mask).flatten()
        for topo_i in downscale.topography_normalise.values():
            topo_i = topo_i.flatten()
            topo_i = topo_i[unmask]
            for i in range(self.area_unmask):
                for j in range(i+1, self.area_unmask):
                    self.square_error[i,j] += math.pow(topo_i[i] - topo_i[j], 2)
                    self.square_error[j,i] = self.square_error[i,j]
    
    def get_n_dim(self):
        return 1
    
    def get_state(self):
        return self.precision
    
    def update_state(self, state):
        self.precision = state
    
    def simulate_from_prior(self, rng):
        self.prior_precision.random_state = rng
        return self.prior_precision.rvs()
    
    def get_cov_chol(self):
        cov_chol = self.square_error.copy()
        cov_chol *= -self.precision/2
        cov_chol = np.exp(cov_chol)
        cov_chol = cholesky(cov_chol)
        return cov_chol

def get_gp_precision_prior():
    return stats.gamma(a=0.72, loc=2.27, scale=8.1)
