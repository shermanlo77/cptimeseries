import math

import numpy as np
from numpy import linalg

from compound_poisson.mcmc import mcmc_abstract

class Rwmh(mcmc_abstract.Mcmc):
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

    def __init__(self, n_sample, memmap_path, target, rng):
        super().__init__(np.float64, n_sample, memmap_path, target, rng)
        self.n_till_adapt = 2*self.n_dim
        self.prob_small_proposal = 0.05
        self.proposal_covariance_small = 1e-4 / self.n_dim
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
        except(ValueError, OverflowError, linalg.LinAlgError):
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

class Elliptical(mcmc_abstract.Mcmc):
    """Elliptical slice sampling

    Elliptical slice sampling, see Murray, Adams, MacKay (2010). Samples from a
        Gaussian prior and do slice sampling

    For more attributes, see the superclass
    Attributes:
        n_reject_array: number of times a rejection was done for each sample
    """

    def __init__(self, n_sample, memmap_path, target, rng):
        super().__init__(np.float64, n_sample, memmap_path, target, rng)
        self.n_reject_array = []

    def sample(self):
        """Uses elliptical slice sampling

        Implemented
        See Murray, Adams, MacKay (2010)
        """
        target = self.target
        state_before = self.state.copy()
        #sample from the prior (with zero mean for now)
        prior_sample = self.simulate_from_prior()
        #sample the vertical line
        ln_y = self.get_log_likelihood() + math.log(self.rng.rand())
        #sample from where on the ellipse
        theta = self.rng.uniform(0, 2 * math.pi)
        edges = [theta - 2 * math.pi, theta]

        #keep sampling until one is accepted
        is_sampling = True
        n_reject = 0
        while is_sampling:
            #get a sample (theta = 0 would just sample itself)
            #centre the parameter at zero mean (relative to the prior)
            self.state = ((state_before - target.get_prior_mean())
                * math.cos(theta) + prior_sample * math.sin(theta))
            #re-centre the parameter
            self.state += target.get_prior_mean()
            #set the new proposed parameter
            #attempt to update all the parameters, reject if there are any
                #numerical problems or when the log likelihood is not large
                #enough
            try:
                self.update_state()
                if self.get_log_likelihood() > ln_y:
                    is_accept = True
                else:
                    is_accept = False
            except(ValueError, OverflowError):
                is_accept = False
            #stop sampling if to accept the new proposed parameter
            #change the search space if the new proposed parameter was rejected
            if is_accept:
                is_sampling = False
            else:
                if theta < 0:
                    edges[0] = theta
                else:
                    edges[1] = theta
                theta =  self.rng.uniform(edges[0], edges[1])
                n_reject += 1
        self.n_reject_array.append(n_reject)

    def simulate_from_prior(self):
        """Return a parameter sampled from the prior

        Must be a Gaussian prior
        """
        return self.target.simulate_from_prior(self.rng)
