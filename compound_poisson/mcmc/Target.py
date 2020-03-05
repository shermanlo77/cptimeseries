import numpy as np
import scipy.stats as stats

import compound_poisson as cp

class Target:
    """Abstract class for evaluating the posterior distribution for MCMC
    
    Subclasses will need to implement the methods get_n_dim(), get_state(),
        update_state(), get_log_likelihood(), get_log_target(), save_state(),
        revert_state(), simulate_from_prior().
    """
    
    def __init__(self):
        pass
    
    def get_n_dim(self):
        """Return the number of dimensions
        """
        raise NotImplementedError
    
    def get_state(self):
        """Return the state vector from the model
        """
        raise NotImplementedError
    
    def update_state(self, state):
        """Update the model with a new state vector
        """
        raise NotImplementedError
    
    def get_log_likelihood(self):
        """Return the log likelihood of the model
        
        Return the log likelihood of the target using its state. Call
            update_state() to update the target's state.
        """
        raise NotImplementedError
    
    def get_log_target(self):
        """Return the log target distribution (posterior distribution)
        """
        raise NotImplementedError
    
    def save_state(self):
        """Save a copy of the state vector
        """
        raise NotImplementedError
    
    def revert_state(self):
        """Revert the state vector and the model state to how it was before
            save_state() was called
        
        From the state vector saved in save_state, update the model with it
            and set self.state to it. In other words, 
        """
        raise NotImplementedError
    
    def simulate_from_prior(self, rng):
        """Simulate from the prior
        
        Args:
            rng: numpy.random.RandomState object
        
        Returns:
            Sample from the prior
        """
        raise NotImplementedError

def get_parameter_mean_prior(parameter_name_array):
    """Return the default mean prior for the parameter
    
    Args:
        parameter_name_array: array of names of each parameter
    
    Return:
        prior mean for the parameter
    """
    prior_mean = np.zeros(len(parameter_name_array))
    #check the parameter by looking into the parameter name
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
    """Return the default prior std for the parameter
    """
    return 0.5

def get_precision_prior():
    """Return the default prior for the precision
    
    Return:
        array containing 2 gamma distributions, first one for the parameter,
            second for ARMA terms
    """
    prior = [
        stats.gamma(a=2.8, scale=2.3),
        stats.gamma(a=1.3, loc=16, scale=65),
    ]
    return prior

def get_arma_index(parameter_name_array):
    """Check with parameters are ARMA terms
    
    Args:
        parameter_name_array
    
    Return:
        boolean array, True if it is a ARMA parameter
    """
    arma_index = []
    for parameter_name in parameter_name_array:
        if "_AR" in parameter_name:
            arma_index.append(True)
        elif "_MA" in parameter_name:
            arma_index.append(True)
        else:
            arma_index.append(False)
    return arma_index
    
