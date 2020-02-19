import math

class Mcmc:
    
    def __init__(self, target, rng):
        self.target = target
        self.rng = rng
        self.n_dim = self.target.get_n_dim()
        self.state = target.get_state()
        self.sample_array = []
    
    def add_to_sample(self):
        self.sample_array.append(self.state.copy())
    
    def update_state(self):
        self.target.update_state(self.state)
    
    def get_log_likelihood(self):
        return self.target.get_log_likelihood()
    
    def get_log_target(self):
        return self.target.get_log_target()
    
    def save_state(self):
        self.target.save_state()
    
    def revert_state(self):
        self.target.revert_state()
        self.state = self.target.get_state()
    
    def sample(self):
        pass
    
    def append(self, state):
        self.sample_array.append(state)
    
    def step(self):
        self.sample()
        self.append(self.state.copy())
    
    def is_accept_step(self, log_target_before, log_target_after):
        """Metropolis-Hastings accept and reject
        
        Args:
            log_target_before
            log_target_after
        
        Returns:
            True if this is the accept step, else False
        """
        accept_prob = math.exp(log_target_after - log_target_before)
        if self.rng.rand() < accept_prob:
            return True
        else:
            return False
    
    def __len__(self):
        return len(self.sample_array)
    
    def __getitem__(self, index):
        return self.sample_array[index]
