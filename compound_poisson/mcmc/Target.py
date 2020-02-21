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
    
