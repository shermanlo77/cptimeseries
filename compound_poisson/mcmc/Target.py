class Target:
    
    def __init__(self):
        self.prior_mean = None #used by slice sampling
        self.prior_cov_chol = None #used by slice sampling
    
    def get_n_dim(self):
        pass
    
    def get_state(self):
        pass
    
    def update_state(self, state):
        pass
    
    def get_log_likelihood(self):
        pass
    
    def get_log_target(self):
        pass
    
    def save_state(self):
        pass
    
    def revert_state(self):
        pass
    
