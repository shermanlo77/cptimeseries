import math

import numpy as np

from .Mcmc import Mcmc

class Rwmh(Mcmc):
    """Random walk Metropolis Hastings
    
    Adaptive MCMC from Roberts and Rosenthal (2009). Proposal covariance is
        proportional to the sample covariance of the chain.
    
    Attributes:
        n_till_adapt: the chain always use the small proposal initially, number
            of steps till use the adaptive proposal covariance
        prob_small_proposal: probability of using proposal_covariance_small as
            the proposal covariance for the reggression parameters
        proposal_covariance_small: the size of the small proposal covariance,
            scalar, it is to be multipled by an identity matrix
        proposal_scale: proposal covariance for the regression parameters is
            proposal_scale times chain_covariance
        chain_mean: mean of the regression parameter chain (excludes z step)
        chain_covariance: covariance of the regression parameter chain (excludes
            samples from sampling other dimensions)
        n_propose: number of proposals
        n_accept: number of accept steps
        accept_array: acceptance rate at each proposal
    """
    
    def __init__(self, target, rng):
        super().__init__(target, rng)
        self.n_till_adapt = 2*self.n_dim
        self.prob_small_proposal = 0.05
        self.proposal_covariance_small = 1e-8 / self.n_dim
        self.proposal_scale = math.pow(2.38,2) / self.n_dim
        self.chain_mean = self.state.copy()
        self.chain_covariance = np.zeros((self.n_dim, self.n_dim))
        self.n_propose = 0
        self.n_accept = 0
        self.accept_array = []
    
    def sample(self):
        """Use Metropolis Hastings to sample parameters
        
        Implemented
        Normal prior, normal proposal
        """
        #decide to use small proposal or adaptive proposal
        #for the first n_adapt runs, they are small to ensure chain_covariance
            #is full rank
        if self.n_propose < self.n_till_adapt:
            is_small_proposal = True
        #after adapting, mixture of small proposal and adaptive proposal
        else:
            if self.rng.rand() < self.prob_small_proposal:
                is_small_proposal = True
            else:
                is_small_proposal = False
        #set the proposal covariance
        if is_small_proposal:
            proposal_covariance = (self.proposal_covariance_small
                * np.identity(self.n_dim))
        else:
            proposal_covariance = self.proposal_scale * self.chain_covariance
        
        #copy the reg_parameters, in case of rejection step
        self.save_state()
        #get posterior
        log_posterior_before = self.get_log_target()
        #make a step if there are numerical problems, treat it as a rejection
            #step
        try:
            #self.propose(proposal_covariance) changes the all the member
                #variable parameters, the likelihood is evaulated with these new
                #parameters
            self.propose(proposal_covariance)
            log_posterior_after = self.get_log_target()
            if not self.is_accept_step(
                log_posterior_before, log_posterior_after):
                #regression step, set the parameters before the MCMC step
                self.revert_state()
            else:
                #acceptance step, keep track of acceptance rate
                self.n_accept += 1
        except(ValueError, OverflowError):
            #treat numerical problems as a rejection step
            self.revert_state()
        #keep track of acceptance rate
        self.n_propose += 1
        self.accept_array.append(self.n_accept / self.n_propose)
        #update the proposal covariance
        self.update_chain_statistics()
    
    def propose(self, covariance):
        """Propose parameters
        
        Update itself with the proposed regression parameters.
        
        Args:
            covariance: proposal covariance
        """
        self.state = self.rng.multivariate_normal(self.state, covariance)
        self.update_state()
    
    def update_chain_statistics(self):
        """Update the statistics of the parameter chain
        
        Update the chain mean and chain covariance of the parameter
            chain. This is used for the adaptive proposal for the regression
            parameter chain.
        """
        n = self.n_propose + 1
        if n == 2:
            state_0 = self.chain_mean.copy()
        state = self.state
        self.chain_mean *= (n-1) / n
        self.chain_mean += state / n
        diff = state - self.chain_mean
        #if there are 2 data points, start working out the covariance
        if n == 2:
            diff0 = state_0 - self.chain_mean
            self.chain_covariance = np.outer(diff0, diff0)
            self.chain_covariance += np.outer(diff, diff)
        #online update the covariance matrix
        elif n > 2:
            self.chain_covariance *= (n-2) / (n-1)
            self.chain_covariance += (n/math.pow(n-1,2)) * np.outer(diff, diff)
