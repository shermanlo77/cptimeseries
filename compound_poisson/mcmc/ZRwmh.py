import math

import numpy as np

from .Mcmc import Mcmc

class ZRwmh(Mcmc):
    
    def __init__(self, target, rng):
        super().__init__(target, rng)
        self.z_parameter = 1 / target.get_n_dim()
        self.n_propose = 0
        self.n_accept = 0
        self.accept_array = []
    
    def sample(self):
        """Use Metropolis Hastings to sample z
        
        Uniform prior, proposal is step back, stay or step forward
        """
        #make a deep copy of the z, use it in case of rejection step
        self.save_state()
        state_before = self.state.copy()
        log_target_before = self.get_log_likelihood()
        #count number of times z moves to 1, also subtract if z moves from 1
        #use because the transition is non-symetric at 1
        n_transition_to_one = 0
        #for each z, proposal
        for i in range(len(self.state)):
            z = self.state[i]
            if z != 0:
                self.state[i] = self.propose_z(z)
                if z == 1 and self.state[i] == 2:
                    n_transition_to_one -= 1
                elif z == 2 and self.state[i] == 1:
                    n_transition_to_one += 1
        try:
            #metropolis hastings
            #needed for get_joint_log_likelihood()
            self.update_state()
            log_target_after = self.get_log_likelihood()
            #to cater for non-symetric transition
            log_target_after += n_transition_to_one * math.log(2)
            if not self.is_accept_step(log_target_before, log_target_after):
                #rejection step, replace the z before this MCMC step
                self.revert_state()
            else:
                #accept step only if at least one z has moved
                if not np.array_equal(state_before, self.state):
                    self.n_accept += 1
        #treat numerical errors as a rejection
        except(ValueError, OverflowError):
            self.revert_state()
        #keep track of acceptance rate
        self.n_propose += 1
        self.accept_array.append(self.n_accept/ self.n_propose)
    
    def propose_z(self, z):
        """Return a proposed z
        
        Return a transitioned z. Probability that it will move is
            self.proposal_z_parameter, otherwise move one step. Cannot move to
            0.
        
        Args:
            z: z at this step of MCMC
        
        Returns:
            Proposed z
        """
        if self.rng.rand() < self.z_parameter:
            if z == 1: #cannot move to 0, so move to 2
                return 2
            elif self.rng.rand() < 0.5:
                return z+1
            else:
                return z-1
        else:
            return z
