import math
import numpy as np
import numpy.random as random

from TimeSeries import TimeSeries
from scipy.stats import multivariate_normal

class TimeSeriesMcmc(TimeSeries):
    """Fit Compound Poisson time series using Bayesian setting
    
    Method uses Metropolis Hastings within Gibbs. Sample either the z or the
        regression parameters. Uniform prior on z, Normal prior on the
        regression parameters. Adaptive MCMC from Roberts and Rosenthal (2009)
    
    Attributes:
        n_sample: number of MCMC samples
        z_sample: array of z samples
        parameter_sample: array of regression parameters in vector form
        proposal_z_parameter: for proposal,probability of movingq z
        prior_mean: prior mean for the regression parameters
        prior_covariance: prior covariance for the regression parameters
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
            z step)
        n_propose_z: number of z proposals
        n_propose_reg: number of proposals for the regression parameters
        n_accept_z: number of accept steps when sampling z
        n_accept_reg: number of accept steps when sampling the regression
            parameters
        accept_reg_array: acceptance rate for the regression parameter chain
        accept_z_array: acceptance rate for the z chain
        rng: random number generator
    """
    
    def __init__(self, x, cp_parameter_array):
        super().__init__(x, cp_parameter_array)
        self.n_sample = 92000
        self.z_sample = []
        self.parameter_sample = []
        self.proposal_z_parameter = 1/self.n
        self.prior_mean = np.zeros(self.n_parameter)
        self.prior_covariance = 0.25*np.identity(self.n_parameter)
        self.n_till_adapt = 2*self.n_parameter
        self.prob_small_proposal = 0.05
        self.proposal_covariance_small = 1e-8 / self.n_parameter
        self.proposal_scale = math.pow(2.38,2)/self.n_parameter
        self.chain_mean = np.zeros(self.n_parameter)
        self.chain_covariance = np.zeros((self.n_parameter, self.n_parameter))
        self.n_propose_z = 0
        self.n_propose_reg = 0
        self.n_accept_z = 0
        self.n_accept_reg = 0
        self.accept_reg_array = []
        self.accept_z_array = []
        self.rng = random.RandomState(np.uint32(2057577976))
    
    def fit(self):
        """Do MCMC
        """
        #set the prior mean to be the initial value
        self.prior_mean = self.get_parameter_vector()
        self.e_step() #initalise the z using the E step
        self.z_array = self.z_array.round() #round it to get integer
        #z cannot be 0 if y is not 0
        self.z_array[np.logical_and(self.z_array==0, self.y_array>0)] = 1
        self.update_all_cp_parameters() #initalse cp parameters
        #initial value is a sample
        self.z_sample.append(self.z_array.copy())
        self.parameter_sample.append(self.get_parameter_vector())
        #use initial value to initalise the adaptive mcmc
        self.update_chain_statistics()
        #Gibbs sampling
        for i in range(self.n_sample):
            print("Sample",i)
            #select random component
            if self.rng.rand() < 0.5:
                self.metropolis_hastings_on_z()
            else:
                self.metropolis_hastings_on_reg()
            #add parameter to the array of samples
            self.z_sample.append(self.z_array.copy())
            self.parameter_sample.append(self.get_parameter_vector())
        #set the parameter the posterior mean
        self.update_all_cp_parameters()
        self.set_parameter_vector(self.chain_mean)
    
    def metropolis_hastings_on_z(self):
        """Use Metropolis Hastings to sample z
        
        Uniform prior, proposal is step back, stay or step forward
        """
        #make a deep copy of the z, use it in case of rejection step
        z_before = self.z_array.copy()
        cp_parameter_before = self.copy_parameter()
        log_posterior_before = self.get_joint_log_likelihood()
        #count number of times z moves to 1, also subtract if z moves from 1
        #use because the transition is non-symetric at 1
        n_transition_to_one = 0
        #for each z, proposal
        for i in range(self.n):
            z = self.z_array[i]
            if z != 0:
                self.z_array[i] = self.propose_z(z)
                if z == 1 and self.z_array[i] == 2:
                    n_transition_to_one -= 1
                elif z == 2 and self.z_array[i] == 1:
                    n_transition_to_one += 1
        try:
            #metropolis hastings
            #needed for get_joint_log_likelihood()
            self.update_all_cp_parameters()
            log_posterior_after = self.get_joint_log_likelihood()
            #to cater for non-symetric transition
            log_posterior_after += n_transition_to_one * math.log(2)
            if not self.is_accept_step(log_posterior_before,
                log_posterior_after):
                #rejection step, replace the z before this MCMC step
                self.z_array = z_before
                self.set_parameter(cp_parameter_before)
            else:
                #accept step only if at least one z has moved
                if not np.array_equal(z_before, self.z_array):
                    self.n_accept_z += 1
        #treat numerical errors as a rejection
        except(ValueError, OverflowError):
            self.z_array = z_before
            self.set_parameter(cp_parameter_before)
        #keep track of acceptance rate
        self.n_propose_z += 1
        self.accept_z_array.append(self.n_accept_z/self.n_propose_z)
    
    def metropolis_hastings_on_reg(self):
        """Use Metropolis Hastings to sample regression parameters
        
        Normal prior, normal proposal
        """
        #decide to use small proposal or adaptive proposal
        #for the first n_adapt runs, they are small to ensure chain_covariance
            #is full rank
        if self.n_propose_reg < self.n_till_adapt:
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
                * np.identity(self.n_parameter))
        else:
            proposal_covariance = self.proposal_scale * self.chain_covariance
        
        #copy the reg_parameters, in case of rejection step
        cp_parameter_before = self.copy_parameter()
        #get posterior
        log_posterior_before = (self.get_joint_log_likelihood()
            + self.get_log_prior())
        #make a step, self.propose(proposal_covariance) returns False if there
            #are numerical problems, treat it as a rejection step
        if self.propose_reg(proposal_covariance):
            #self.propose(proposal_covariance) changes the all the member
                #variable parameters, the likelihood is evaulated with these new
                #parameters
            log_posterior_after = (self.get_joint_log_likelihood()
                + self.get_log_prior())
            if not self.is_accept_step(
                log_posterior_before, log_posterior_after):
                #regression step, set the parameters before the MCMC step
                self.set_parameter(cp_parameter_before)
            else:
                #acceptance step, keep track of acceptance rate
                self.n_accept_reg += 1
        else:
            #treat numerical problems as a rejection step
            self.set_parameter(cp_parameter_before)
        #keep track of acceptance rate
        self.n_propose_reg += 1
        self.accept_reg_array.append(self.n_accept_reg/self.n_propose_reg)
        #update the proposal covariance
        self.update_chain_statistics()
    
    def is_accept_step(self, log_posterior_before, log_posterior_after):
        """Metropolis-Hastings accept and reject
        
        Args:
            log_posterior_before
            log_posterior_after
        
        Returns:
            True if this is the accept step, else False
        """
        accept_prob = math.exp(log_posterior_after - log_posterior_before)
        if self.rng.rand() < accept_prob:
            return True
        else:
            return False
    
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
        if self.rng.rand() < self.proposal_z_parameter:
            if z == 1: #cannot move to 0, so move to 2
                return 2
            elif self.rng.rand() < 0.5:
                return z+1
            else:
                return z-1
        else:
            return z
    
    def get_log_prior(self):
        """Log prior for the regression parameter
        
        Uses the current (member variable) parameter
        """
        return multivariate_normal.logpdf(
            self.get_parameter_vector(), self.prior_mean, self.prior_covariance)
    
    def propose_reg(self, covariance):
        """Propose regression parameters
        
        Update itself with the proposed regression parameters.
        
        Args:
            covariance: proposal covariance
        
        Returns:
            False if there are numerical problems, otherwise True
        """
        parameter_vector = self.get_parameter_vector()
        parameter_vector = self.rng.multivariate_normal(
            parameter_vector, covariance)
        #update it's member variables with the proposed parameter
        self.set_parameter_vector(parameter_vector)
        try:
            self.update_all_cp_parameters()
            return True
        except (ValueError, OverflowError):
            return False
    
    def update_chain_statistics(self):
        """Update the statistics of the regression parameter chain
        
        Update the chain mean and chain covariance of the regression parameter
            chain. This is used for the adaptive proposal for the regression
            parameter chain.
        """
        n = len(self.parameter_sample)
        parameter = self.parameter_sample[n-1]
        self.chain_mean *= (n-1)/n
        self.chain_mean += parameter / n
        diff = parameter-self.chain_mean
        #if there are 2 data points, start working out the covariance
        if n == 2:
            diff0 = parameter[0]-self.chain_mean
            self.chain_covariance = np.outer(diff0, diff0)
            self.chain_covariance += np.outer(diff, diff)
        #online update the covariance matrix
        elif n>2:
            self.chain_covariance *= (n-2)/(n-1)
            self.chain_covariance += (n/math.pow(n-1,2)) * np.outer(diff, diff)
