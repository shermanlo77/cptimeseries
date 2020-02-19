import math
import numpy as np
import numpy.random as random
from scipy.stats import norm

from .TimeSeries import TimeSeries
from .mcmc import Rwmh, TargetParameter, TargetZ, ZRwmh

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
        self.n_sample = 100000
        self.parameter_mcmc = None
        self.z_mcmc = None
    
    def fit(self):
        """Do MCMC
        """
        self.initalise_z()
        self.instantiate_mcmc()
        #initial value is a sample
        self.parameter_mcmc.add_to_sample()
        self.z_mcmc.add_to_sample()
        #Gibbs sampling
        for i in range(self.n_sample):
            print("Sample",i)
            #select random component
            if self.rng.rand() < 0.5:
                self.z_mcmc.step()
                self.parameter_mcmc.add_to_sample()
            else:
                self.parameter_mcmc.step()
                self.z_mcmc.add_to_sample()
    
    def initalise_z(self):
        self.e_step() #initalise the z using the E step
        self.z_array = self.z_array.round() #round it to get integer
        #z cannot be 0 if y is not 0
        self.z_array[np.logical_and(self.z_array==0, self.y_array>0)] = 1
        self.update_all_cp_parameters() #initalse cp parameters
    
    def instantiate_mcmc(self):
        self.parameter_mcmc = Rwmh(TargetParameter(self), self.rng)
        self.z_mcmc = ZRwmh(TargetZ(self), self.rng)
    
    def set_parameter_from_sample(self, index):
        """Set parameter from MCMC sample
        
        Set the regression parameters and latent variables z from the MCMC
            samples in parameter_sample and z_sample. NOTE: does a shallow copy
            from the array of samples.
        """
        self.set_parameter_vector(self.parameter_mcmc[index])
        self.z_array = self.z_mcmc[index]
        self.update_all_cp_parameters()
    
    def instantiate_forecast_self(self):
        """Override - Set the parameter from the MCMC sample
        """
        self.set_parameter_from_sample(
            self.rng.randint(self.burn_in, self.n_sample))
        forecast = super().instantiate_forecast_self()
        return forecast
    
    def instantiate_forecast(self, x):
        """Override - Set the parameter from the MCMC sample
        """
        self.set_parameter_from_sample(
            self.rng.randint(self.burn_in, self.n_sample))
        forecast = super().instantiate_forecast(x)
        return forecast
    
    def simulate_parameter_from_prior(self):
        """Return a parameter sampled from the prior
        """
        parameter = np.empty(self.n_parameter)
        for i in range(self.n_parameter):
            parameter[i] = self.rng.normal(
                self.prior_mean[i], math.sqrt(self.prior_covariance[i]))
        return parameter
    
    def simulate_from_prior(self):
        """Simulate using a parameter from the prior
        
        Modifies itself, the prior mean and prior covariance unmodified
        """
        while True:
            try:
                self.set_parameter_vector(self.simulate_parameter_from_prior())
                self.simulate()
                #check if any of the parameters are not nan
                if np.any(np.isnan(self.poisson_rate.value_array)):
                    pass
                elif np.any(np.isnan(self.gamma_mean.value_array)):
                    pass
                elif np.any(np.isnan(self.gamma_dispersion.value_array)):
                    pass
                else:
                    break
            except(ValueError, OverflowError):
                pass
