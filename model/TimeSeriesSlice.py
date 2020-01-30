import math
import numpy as np

from TimeSeriesMcmc import TimeSeriesMcmc

class TimeSeriesSlice(TimeSeriesMcmc):
    
    def __init__(self, 
                 x,
                 rainfall=None,
                 poisson_rate_n_arma=None,
                 gamma_mean_n_arma=None,
                 cp_parameter_array=None):
        super().__init__(x,
                         rainfall,
                         poisson_rate_n_arma,
                         gamma_mean_n_arma,
                         cp_parameter_array)
        self.non_zero_index = np.nonzero(self.y_array)[0]
    
    def sample_z(self):
        """Override - Use slice sampling
        
        Select a random z_t in the time series, then move it using slice
            sampling
        """
        #pick a random non-zero latent variable to sample
        t = self.rng.choice(self.non_zero_index)
        #get the latent variable and the log likelihood
        z_t = self.z_array[t]
        ln_l_before = self.get_joint_log_likelihood()
        #sample the vertical line
        ln_y = math.log(self.rng.rand()) + ln_l_before
        #proposal range contains the limits of what integers to propose, both
            #excluse for now
        proposal_range = [z_t-1, z_t+1]
        
        #keep decreasing proposal_range[0] until it is zero or when the log
            #likelihood is less than ln_y
        #NOTE: proposal_range is exclusive for now
        ln_l_propose = ln_l_before
        while ln_l_propose > ln_y:
            if proposal_range[0] == 0:
                break
            else:
                #break when there is a numerical problem with this proposal
                try:
                    self.z_array[t] = proposal_range[0]
                    self.update_all_cp_parameters()
                    ln_l_propose = self.get_joint_log_likelihood()
                    proposal_range[0] -= 1
                except(ValueError, OverflowError):
                    break
        #NOTE: now make proposal range left inclusive
        proposal_range[0] += 1
        
        #keep increasing proposal_range[1] until the log likelihood is less than
            #ln_y
        #NOTE: proposal_range is left inclusive and right exclusive for the rest
            #of this method
        ln_l_propose = ln_l_before
        while ln_l_propose > ln_y:
            #break when there is a numerical problem with this proposal
            try:
                self.z_array[t] = proposal_range[1]
                self.update_all_cp_parameters()
                ln_l_propose = self.get_joint_log_likelihood()
                proposal_range[1] += 1
            except(ValueError, OverflowError):
                break
        
        #set the proposed z
        self.z_array[t] = self.rng.randint(proposal_range[0], proposal_range[1])
        self.update_all_cp_parameters()
