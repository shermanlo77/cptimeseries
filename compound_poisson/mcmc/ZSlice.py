import math

import numpy as np

from .Mcmc import Mcmc

class ZSlice(Mcmc):
    
    def __init__(self, target, rng):
        super().__init__(target, rng)
        self.n_propose = 0
        self.slice_width_array = []
        self.non_zero_index = np.nonzero(self.target.time_series.y_array)[0]
    
    def sample(self):
        """Implemented - Use slice sampling
        
        See Neal (2003)
        Select a random z_t in the time series, then move it using slice
            samplingls
        """
        time_series = self.target.time_series
        #pick a random non-zero latent variable to sample
        t = self.rng.choice(self.non_zero_index)
        #get the latent variable and the log likelihood
        z_t = self.state[t]
        ln_l_before = self.get_log_likelihood()
        #sample the vertical line
        ln_y = math.log(self.rng.rand()) + ln_l_before
        #proposal range contains the limits of what integers to propose, both
            #excluse for now
        proposal_range = [z_t, z_t+1]
        
        #keep decreasing proposal_range[0] until it is zero or when the log
            #likelihood is less than ln_y
        #NOTE: proposal_range is exclusive for now
        ln_l_propose = ln_l_before
        is_in_slice = True
        while is_in_slice:
            proposal_range[0] -= 1
            if proposal_range[0] == 0:
                is_in_slice = False
            else:
                #break when there is a numerical problem with this proposal
                try:
                    self.state[t] = proposal_range[0]
                    self.update_state()
                    ln_l_propose = self.get_log_likelihood()
                    if ln_l_propose < ln_y:
                        is_in_slice = False
                except(ValueError, OverflowError):
                    is_in_slice = False
        #NOTE: now make proposal range left inclusive
        proposal_range[0] += 1
        
        #keep increasing proposal_range[1] until the log likelihood is less than
            #ln_y
        #NOTE: proposal_range is left inclusive and right exclusive for the rest
            #of this method
        ln_l_propose = ln_l_before
        is_in_slice = True
        while is_in_slice:
            #break when there is a numerical problem with this proposal
            try:
                self.state[t] = proposal_range[1]
                self.update_state()
                ln_l_propose = self.get_log_likelihood()
                if ln_l_propose < ln_y:
                    is_in_slice = False
                else:
                    proposal_range[1] += 1
            except(ValueError, OverflowError):
                is_in_slice = False
        
        #set the proposed z
        self.state[t] = self.rng.randint(proposal_range[0], proposal_range[1])
        self.update_state()
        self.slice_width_array.append(proposal_range[1] - proposal_range[0])
        self.n_propose += 1
