import math

import numpy as np

from compound_poisson import mcmc
from compound_poisson import time_series
from compound_poisson.mcmc import target_time_series

class TimeSeriesMcmc(time_series.TimeSeries):
    """Fit Compound Poisson time series using Rwmh from a Bayesian setting

    Method uses Metropolis Hastings within Gibbs. Sample either the z or the
        regression parameters. Uniform prior on z, Normal prior on the
        regression parameters. Adaptive MCMC from Roberts and Rosenthal (2009)

    For more attributes, see the superclass
    Attributes:
        n_sample: number of MCMC samples
        parameter_target: wrapper Target object to evaluate the posterior of the
            parameters
        parameter_mcmc: Mcmc object which does MCMC using parameter_target
        z_target: wrapper Target object to evaluate the posterior of z
        z_mcmc: Mcmc object which does MCMC using z_target
        burn_in: integer, which samples to discard when doing posterior sampling
            which is used for forecasting
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
        self.parameter_target = target_time_series.TargetParameter(self)
        self.parameter_mcmc = None
        self.z_target = target_time_series.TargetZ(self)
        self.z_mcmc = None
        self.burn_in = 0

    def fit(self):
        """Fit using Gibbs sampling

        Override
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
            rand = self.rng.rand()
            if rand < 0.5:
                self.z_mcmc.step()
                self.parameter_mcmc.add_to_sample()
            else:
                self.parameter_mcmc.step()
                self.z_mcmc.add_to_sample()

    def initalise_z(self):
        """Initalise all z in self.z_array and update all parameters

        Initalise all z in self.z_array using e_step() and update all parameters
            using update_all_cp_parameters(). Required for e.g. likelihood
            evaluation because z=0 if and only if y=0.
        """
        self.e_step() #initalise the z using the E step
        self.z_array = self.z_array.round() #round it to get integer
        #z cannot be 0 if y is not 0
        self.z_array[np.logical_and(self.z_array==0, self.y_array>0)] = 1
        self.update_all_cp_parameters() #initalse cp parameters

    def instantiate_mcmc(self):
        """Instantiate all MCMC objects

        Instantiate all MCMC objects by passing the corresponding Target objects
            and random number generators
        """
        self.parameter_mcmc = mcmc.Rwmh(self.parameter_target, self.rng)
        self.z_mcmc = mcmc.ZRwmh(self.z_target, self.rng)

    def set_parameter_from_sample(self):
        """Set parameter from MCMC sample

        Set the regression parameters and latent variables z from the MCMC
            samples in parameter_sample and z_sample.
        """
        index = self.rng.randint(self.burn_in, len(self.parameter_mcmc))
        self.set_parameter_vector(self.parameter_mcmc[index])
        self.z_array = self.z_mcmc[index]
        self.update_all_cp_parameters()

    def instantiate_forecast_self(self):
        """Override - Set the parameter from the MCMC sample
        """
        self.set_parameter_from_sample()
        forecast = super().instantiate_forecast_self()
        return forecast

    def instantiate_forecast(self, x):
        """Override - Set the parameter from the MCMC sample
        """
        self.set_parameter_from_sample()
        forecast = super().instantiate_forecast(x)
        return forecast

    def simulate_from_prior(self):
        """Simulate using a parameter from the prior

        MODIFIES ITSELF
        Replaces the parameter with a sample from the prior. The prior mean and
            prior covariance unmodified.
        """
        #keep sampling until the sampled parameter does not have numerical
            #problems
        while True:
            try:
                #sample from the prior and set it
                prior_parameter = self.simulate_parameter_from_prior()
                self.set_parameter_vector(prior_parameter)
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
            #try again if there are numerical problems
            except(ValueError, OverflowError):
                pass

    def simulate_parameter_from_prior(self):
        """Simulate parameter from the prior

        Return a sample from the prior
        """
        return self.parameter_target.simulate_from_prior(self.rng)

class TimeSeriesSlice(TimeSeriesMcmc):
    """Fit Compound Poisson time series using slice sampling from a Bayesian
    setting

    Method uses slice within Gibbs. Sample either the z or the
        regression parameters. Uniform prior on z, Normal prior on the
        regression parameters. Sample z using slice sampling (Neal, 2003).
        Sampling the parameters using elliptical slice sampling (Murray 2010).

    For more attributes, see the superclass
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
        self.n_sample = 10000

    def instantiate_mcmc(self):
        """Instantiate all MCMC objects

        Override
        Instantiate slice sampling for the parameter and z
        """
        self.parameter_mcmc = mcmc.Elliptical(self.parameter_target, self.rng)
        self.z_mcmc = mcmc.ZSlice(self.z_target, self.rng)

class TimeSeriesHyperSlice(TimeSeriesSlice):
    """Fit Compound Poisson time series using slice sampling with a prior on the
    precision

    Method uses slice within Gibbs. Uniform prior on z, Normal prior on the
        regression parameters, Gamma prior on the precision of the covariance of
        the Normal prior. Gibbs sample either z, regression parameters or the
        precision. Sample z using slice sampling (Neal, 2003). Sampling the
        parameters using elliptical slice sampling (Murray 2010).

    For more attributes, see the superclass
    Attributes:
        precision_target: wrapper Target object with evaluates the posterior of
            the precision
        precision_mcmc: Mcmc object for precision_target
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
        self.precision_target = target_time_series.TargetPrecision(
            self.parameter_target)
        #mcmc object evaluated at instantiate_mcmc
        self.precision_mcmc = None

    def fit(self):
        """Fit using Gibbs sampling

        Override - Gibbs sample either z, regression parameters or the
            precision.
        """
        self.initalise_z()
        self.instantiate_mcmc()
        self.update_precision()
        #initial value is a sample
        self.parameter_mcmc.add_to_sample()
        self.z_mcmc.add_to_sample()
        self.precision_mcmc.add_to_sample()
        #Gibbs sampling
        for i in range(self.n_sample):
            print("Sample",i)
            #select random component
            rand = self.rng.rand()
            if rand < 1/3:
                self.z_mcmc.step()
                self.parameter_mcmc.add_to_sample()
                self.precision_mcmc.add_to_sample()
            elif rand < 2/3:
                self.parameter_mcmc.step()
                self.z_mcmc.add_to_sample()
                self.precision_mcmc.add_to_sample()
            else:
                self.precision_mcmc.step()
                self.parameter_mcmc.add_to_sample()
                self.z_mcmc.add_to_sample()
                #update the prior covariance in parameter_target
                self.update_precision()

    def update_precision(self):
        """Propagate the precision in precision_target to parameter_target
        """
        self.parameter_target.prior_cov_chol = (
            self.precision_target.get_cov_chol())

    def instantiate_mcmc(self):
        """Instantiate all MCMC objects

        Override - instantiate the MCMC for the precision
        """
        super().instantiate_mcmc()
        self.precision_mcmc = mcmc.Rwmh(self.precision_target, self.rng)
        self.precision_mcmc.proposal_covariance_small = 1e-4

    def simulate_parameter_from_prior(self):
        """Simulate parameter from the prior

        Override - Sample the precision from the prior and then sample the
            parameter from the prior
        """
        prior_precision = self.precision_target.simulate_from_prior(self.rng)
        self.update_precision()
        return super().simulate_parameter_from_prior()
