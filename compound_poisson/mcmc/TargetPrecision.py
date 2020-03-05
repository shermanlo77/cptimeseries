import numpy as np

from .Target import Target
from .Target import get_arma_index
from .Target import get_precision_prior

class TargetPrecision(Target):
    """Wrapper Target class to evaluate the posterior of the precision of the
        parameter prior covariance
    
    Attributes:
        parameter_target: TargetParameter object
        prior: array containing 2 distributions, random_state are accessed,
            rvs() are logpdf() are called. E.g. use distributions from
            scipy.stats. 0th prior for regression parameter, 1st for ARMA
            parameter
        precision: 2 vector
        precision_before: copy of precision
        arma_index: array of boolean, pointing to each parameter, True if it the
            parameter is an ARMA term 
    """
    
    def __init__(self, parameter_target):
        super().__init__()
        self.parameter_target = parameter_target
        self.prior = get_precision_prior()
        self.precision = [] 
        self.precision_before = None
        self.arma_index = None
        
        #initalise precision using the prior mean
        for prior in self.prior:
            self.precision.append(prior.mean())
        self.precision = np.asarray(self.precision)
        
        #initalise arma_index
        time_series = self.parameter_target.time_series
        self.arma_index = get_arma_index(
            time_series.get_parameter_vector_name())
    
    def get_cov_chol(self):
        """Return the vector of parameter prior std
        
        Returns:
            vector, element of each parameter, prior std
        """
        cov_chol = []
        for is_arma in self.arma_index:
            if is_arma:
                cov_chol.append(1 / self.precision[1])
            else:
                cov_chol.append(1 / self.precision[0])
        cov_chol = np.sqrt(np.asarray(cov_chol))
        return cov_chol
    
    def get_n_dim(self):
        return len(self.precision)
    
    def get_state(self):
        return self.precision
    
    def update_state(self, state):
        self.precision = state.copy()
    
    def get_log_likelihood(self):
        return self.parameter_target.get_log_prior(self.get_cov_chol())
    
    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()
    
    def get_log_prior(self):
        ln_prior = 0
        for i, prior in enumerate(self.prior):
            ln_prior += prior.logpdf(self.precision[i])
        return ln_prior
    
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
