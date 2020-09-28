import math
from os import path
import pathlib

import matplotlib.pyplot as plt
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
        gibbs_weight: up to a constant, probability of sampling each mcmc in
            self.get_mcmc_array()
        burn_in: integer, which samples to discard when doing posterior sampling
            which is used for forecasting
        memmap_dir: directory to store the MCMC samples
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
        self.gibbs_weight = [0.003*len(self), 1]
        self.burn_in = 0
        self.memmap_dir = ""

    def fit(self):
        """Fit using Gibbs sampling

        Override - Gibbs sample either z, regression parameters or the
            precision.
        """
        self.initalise_z()
        self.instantiate_mcmc()
        mcmc_array = self.get_mcmc_array()
        mcmc.do_gibbs_sampling(
            mcmc_array, self.n_sample, self.rng, self.gibbs_weight)

    def resume_fitting(self, n_sample):
        """Run more MCMC samples

        Args:
            n_sample: new number of mcmc samples
        """
        if n_sample > self.n_sample:
            mcmc_array = self.get_mcmc_array()
            for mcmc_i in mcmc_array:
                mcmc_i.extend_memmap(n_sample)
            #in resume, do not use initial value as sample (False in arg 3)
            mcmc.do_gibbs_sampling(
                mcmc_array, n_sample - self.n_sample, self.rng,
                self.gibbs_weight, False)
            self.n_sample = n_sample
            self.delete_old_memmap()

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
        self.parameter_mcmc = mcmc.Rwmh(
            self.parameter_target, self.rng, self.n_sample, self.memmap_dir)
        self.z_mcmc = mcmc.ZRwmh(
            self.z_target, self.rng, self.n_sample, self.memmap_dir)

    def get_mcmc_array(self):
        """Return array of Mcmc objects

        Each element in this array can be called to do a Gibbs step for
            different components
        """
        mcmc_array = [
            self.z_mcmc,
            self.parameter_mcmc,
        ]
        return mcmc_array

    def set_burn_in(self, burn_in):
        self.burn_in = burn_in

    def set_parameter_from_sample(self, rng):
        """Set parameter from MCMC sample

        Set the regression parameters and latent variables z from the MCMC
            samples in parameter_sample and z_sample.
        """
        index = rng.randint(self.burn_in, len(self.parameter_mcmc))
        self.set_parameter_from_sample_i(index)
        self.update_all_cp_parameters()

    def set_parameter_from_sample_i(self, index):
        """Set parameter for a specified MCMC sample
        """
        self.set_parameter_vector(self.parameter_mcmc[index])
        self.z_array = self.z_mcmc[index]

    def forecast_self(self, n_simulation):
        #override
        self.read_memmap()
        super().forecast_self(n_simulation)
        self.del_memmap()

    def instantiate_forecast_self(self):
        """Override - Set the parameter from the MCMC sample
        """
        self.set_parameter_from_sample(self.self_forecaster_rng)
        forecast = super().instantiate_forecast_self()
        return forecast

    def forecast(self, x, n_simulation):
        #override
        self.read_memmap()
        super().forecast(n_simulation)
        self.del_memmap()

    def instantiate_forecast(self, x):
        """Override - Set the parameter from the MCMC sample
        """
        self.set_parameter_from_sample(self.forecaster_rng)
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

    def read_memmap(self):
        """Read all memmap file handling from all MCMCs
        """
        for mcmc_i in self.get_mcmc_array():
            mcmc_i.read_memmap()

    def del_memmap(self):
        for mcmc_i in self.get_mcmc_array():
            mcmc_i.del_memmap()

    def delete_old_memmap(self):
        for mcmc_i in self.get_mcmc_array():
            mcmc_i.delete_old_memmap()

    def print_mcmc(self, directory, true_parameter=None):
        parameter_name = self.get_parameter_vector_name()
        self.read_memmap()
        chain = np.asarray(self.parameter_mcmc[:])
        for i in range(self.n_parameter):
            chain_i = chain[:,i]
            plt.figure()
            plt.plot(chain_i)
            if not true_parameter is None:
                plt.hlines(true_parameter[i], 0, len(chain)-1)
            plt.ylabel(parameter_name[i])
            plt.xlabel("Sample number")
            plt.savefig(
                path.join(directory, "chain_parameter_" + str(i) + ".pdf"))
            plt.close()

        chain = []
        z_chain = np.asarray(self.z_mcmc[:])
        chain = np.mean(z_chain, 1)
        plt.figure()
        plt.plot(chain)
        plt.ylabel("Mean of latent variables")
        plt.xlabel("Sample number")
        plt.savefig(path.join(directory, "chain_z.pdf"))
        plt.close()

        self.print_chain_property(directory)
        self.del_memmap()

    def print_chain_property(self, directory):
        plt.figure()
        plt.plot(np.asarray(self.parameter_mcmc.accept_array))
        plt.ylabel("Acceptance rate of parameters")
        plt.xlabel("Parameter sample number")
        plt.savefig(path.join(directory, "accept_parameter.pdf"))
        plt.close()

        plt.figure()
        plt.plot(np.asarray(self.z_mcmc.accept_array))
        plt.ylabel("Acceptance self of latent variables")
        plt.xlabel("Latent variable sample number")
        plt.savefig(path.join(directory, "accept_z.pdf"))
        plt.close()

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
        self.parameter_mcmc = mcmc.Elliptical(
            self.parameter_target, self.rng, self.n_sample, self.memmap_dir)
        self.z_mcmc = mcmc.ZSlice(
            self.z_target, self.rng, self.n_sample, self.memmap_dir)

    def print_chain_property(self, directory):
        plt.figure()
        plt.plot(np.asarray(self.parameter_mcmc.n_reject_array))
        plt.ylabel("Number of rejects in parameter slicing")
        plt.xlabel("Parameter sample number")
        plt.savefig(path.join(directory, "n_reject_parameter.pdf"))
        plt.close()

        plt.figure()
        plt.plot(np.asarray(self.z_mcmc.slice_width_array))
        plt.ylabel("Latent variable slice width")
        plt.xlabel("Latent variable sample number")
        plt.savefig(path.join(directory, "slice_width_z.pdf"))
        plt.close()

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
        self.precision_target = target_time_series.TargetPrecision(self)
        #mcmc object evaluated at instantiate_mcmc
        self.precision_mcmc = None
        self.gibbs_weight = [0.003*len(self), 1, 0.2]

    def instantiate_mcmc(self):
        """Instantiate all MCMC objects

        Override - instantiate the MCMC for the precision
        """
        super().instantiate_mcmc()
        self.precision_target.prograte_precision()
        self.precision_mcmc = mcmc.Rwmh(
            self.precision_target, self.rng, self.n_sample, self.memmap_dir)

    def get_mcmc_array(self):
        mcmc_array = super().get_mcmc_array()
        mcmc_array.append(self.precision_mcmc)
        return mcmc_array

    def simulate_parameter_from_prior(self):
        """Simulate parameter from the prior

        Override - Sample the precision from the prior and then sample the
            parameter from the prior
        """
        self.precision_target.set_from_prior(self.rng)
        self.precision_target.prograte_precision()
        return super().simulate_parameter_from_prior()

    def print_chain_property(self, directory):
        super().print_chain_property(directory)
        precision_chain = np.asarray(self.precision_mcmc[:])
        for i in range(2):
            chain_i = precision_chain[:, i]
            plt.figure()
            plt.plot(chain_i)
            plt.ylabel("precision" + str(i))
            plt.xlabel("Sample number")
            plt.savefig(
                path.join(directory, "chain_precision_" + str(i) + ".pdf"))
            plt.close()

        plt.figure()
        plt.plot(np.asarray(self.precision_mcmc.accept_array))
        plt.ylabel("Acceptance rate of parameters")
        plt.xlabel("Parameter sample number")
        plt.savefig(path.join(directory, "accept_precision.pdf"))
        plt.close()

def static_initalise_z(time_series):
    time_series.initalise_z()
    return time_series
