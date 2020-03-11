import math

import numpy as np
from numpy import random

import compound_poisson
from compound_poisson import mcmc
from compound_poisson.mcmc import target_downscale

class Downscale(object):
    """Collection of multiple TimeSeries object
    
    Fit a compound Poisson time series on multiple locations in 2d space.
        Parameters have a Gaussian process (GP) prior, with gamma precision
        prior and gamma GP precision prior.
    
    Attributes:
        n_arma: 2-tuple, containing number of AR and MA terms
        time_series_array: 2d array containing TimeSeries objects
        mask: 2d boolean, True if on water, therefore masked
        parameter_mask_vector: mask as a vector
        n_parameter: number of parameters for one location
        n_total_parameter: n_parameter times number of unmasked time series
        topography: dictonary of topography information
        topography_normalise: dictonary of topography information normalised,
            mean 0, std 1
        shape: 2-tuple, shape of the space
        area: area of the space
        area_unmask: area of the unmasked space
        rng: numpy.random.RandomState object
        parameter_target: TargetParameter object
        precision_mcmc: TargetPrecision object
        gp_target: TargetGp object
        parameter_mcmc: Mcmc object wrapping around parameter_target
        precision_mcmc: Mcmc object wrapping around precision_target
        gp_mcmc: Mcmc object wrapping around gp_target
        n_sample: number of MCMC samples
    """
    
    def __init__(self, data, n_arma=(0,0)):
        self.n_arma = n_arma
        self.time_series_array = []
        self.mask = data.mask
        self.parameter_mask_vector = []
        self.n_parameter = None
        self.n_total_parameter = None
        self.topography = data.topography
        self.topography_normalise = data.topography_normalise
        self.shape = self.mask.shape
        self.area = self.shape[0] * self.shape[1]
        self.area_unmask = np.sum(np.logical_not(self.mask))
        self.rng = random.RandomState()
        self.parameter_target = None
        self.precision_target = None
        self.gp_target = None
        self.parameter_mcmc = None
        self.precision_mcmc = None
        self.gp_mcmc = None
        self.n_sample = 10000
        
        #copy data
        model_field = data.model_field
        rain = data.rain
        time_series_array = self.time_series_array
        
        #instantiate time series for every point in space
        #unmasked points have rain, provide it to the constructor to TimeSeries
        #masked points do not have rain, cannot provide it
        for lat_i in range(self.shape[0]):
            
            time_series_array.append([])
            
            for long_i in range(self.shape[1]):
                
                x_i, rain_i = data.get_data(lat_i, long_i)
                is_mask = self.mask[lat_i, long_i]
                if is_mask:
                    time_series = compound_poisson.TimeSeriesSlice(
                        x_i, poisson_rate_n_arma=n_arma,
                        gamma_mean_n_arma=n_arma)
                else:
                    time_series = compound_poisson.TimeSeriesSlice(
                        x_i, rain_i.data, n_arma, n_arma)
                time_series.id = str(lat_i) + "_" + str(long_i)
                time_series.rng = self.rng
                time_series_array[lat_i].append(time_series)
                for i in range(time_series.n_parameter):
                    self.parameter_mask_vector.append(is_mask)
                self.n_parameter = time_series.n_parameter
        
        self.parameter_mask_vector = np.asarray(self.parameter_mask_vector)
        self.n_total_parameter = self.area_unmask * self.n_parameter
        self.set_target()
    
    def fit(self):
        """Fit using Gibbs sampling
        """
        self.initalise_z()
        self.instantiate_mcmc()
        self.update_reg_gp()
        self.update_reg_precision()
        
        #initial value is a sample
        self.parameter_mcmc.add_to_sample()
        self.precision_mcmc.add_to_sample()
        self.gp_mcmc.add_to_sample()
        self.z_mcmc_add_to_sample()
        
        #Gibbs sampling
        for i in range(self.n_sample):
            print("Sample",i)
            #select random component
            rand = self.rng.rand()
            if rand < 0.25:
                self.z_mcmc_step()
                self.parameter_mcmc.add_to_sample()
                self.precision_mcmc.add_to_sample()
                self.gp_mcmc.add_to_sample()
            elif rand < 0.5:
                self.parameter_mcmc.step()
                self.precision_mcmc.add_to_sample()
                self.gp_mcmc.add_to_sample()
                self.z_mcmc_add_to_sample()
            elif rand < 0.75:
                self.precision_mcmc.step()
                self.update_reg_precision()
                self.parameter_mcmc.add_to_sample()
                self.gp_mcmc.add_to_sample()
                self.z_mcmc_add_to_sample()
            else:
                self.gp_mcmc.step()
                self.update_reg_gp()
                self.parameter_mcmc.add_to_sample()
                self.precision_mcmc.add_to_sample()
                self.z_mcmc_add_to_sample()
    
    def z_mcmc_add_to_sample(self):
        """Duplicate z_array to the z chain
        
        For all time_series, add z_array to the sample_array
        """
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    time_series.z_mcmc.add_to_sample()
    
    def z_mcmc_step(self):
        """All time_series sample z
        """
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    time_series.z_mcmc.step()
    
    def instantiate_mcmc(self):
        """Instantiate MCMC objects
        """
        self.parameter_mcmc = mcmc.Elliptical(self.parameter_target, self.rng)
        self.precision_mcmc = mcmc.Rwmh(self.precision_target, self.rng)
        self.precision_mcmc.proposal_covariance_small = 1e-4
        self.gp_mcmc = mcmc.Rwmh(self.gp_target, self.rng)
        self.gp_mcmc.proposal_covariance_small = 1e-4
        #all time series objects instantiate mcmc objects to store the z chain
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    time_series.instantiate_mcmc()
    
    def initalise_z(self):
        """Initalise z for all time series
        """
        #all time series initalise z, needed so that the likelihood can be
            #evaluated, eg y=0 if and only if x=0
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    time_series.initalise_z()
        self.update_all_cp_parameters()
    
    def set_target(self):
        """Instantiate target objects
        """
        self.parameter_target = target_downscale.TargetParameter(self)
        self.precision_target = target_downscale.TargetPrecision(
            self.parameter_target)
        self.gp_target = target_downscale.TargetGp(self)
    
    def set_rng(self, rng):
        """Set rng for all time series
        """
        self.rng = rng
        for time_series_i in self.time_series_array:
            for time_series in time_series_i:
                time_series.rng = rng
    
    def simulate_i(self, i):
        """Simulate a point in time for all time series
        """
        for time_series_i in self.time_series_array:
            for time_series in time_series_i:
                time_series.simulate_i(i)
    
    def simulate(self):
        """Simulate the entire time series for all time series
        """
        for time_series_i in self.time_series_array:
            for time_series in time_series_i:
                time_series.simulate()
    
    def get_parameter_3d(self):
        """Return the parameters from all time series (3D)
        
        Return a 3D array of all the parameters in all unmasked time series
        """
        parameter = []
        for time_series_lat in self.time_series_array:
            parameter.append([])
            for time_series_i in time_series_lat:
                parameter[-1].append(time_series_i.get_parameter_vector())
        parameter = np.asarray(parameter)
        return parameter
    
    def get_parameter_vector(self):
        """Return the parameters from all time series (vector)
        
        Return a vector of all the parameters in all unmasked time series. The
            vector is arrange in a format suitable for block diagonal Gaussian
            process
        """
        parameter_3d = self.get_parameter_3d()
        parameter_vector = []
        for i in range(self.n_parameter):
            parameter_vector.append(
                parameter_3d[np.logical_not(self.mask), i].flatten())
        return np.asarray(parameter_vector).flatten()
    
    def set_parameter_vector(self, parameter_vector):
        """Set the parameter for each time series
        
        Args:
            parameter_vector: vector of length area_unmask * n_parameter. See
                get_parameter_vector for the format of parameter_vector
        """
        parameter_3d = np.empty(
            (self.shape[0], self.shape[1], self.n_parameter))
        for i in range(self.n_parameter):
            parameter_i = parameter_vector[
                i*self.area_unmask : (i+1)*self.area_unmask]
            parameter_3d[np.logical_not(self.mask), i] = parameter_i
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    parameter_i = parameter_3d[lat_i, long_i, :]
                    self.time_series_array[lat_i][long_i].set_parameter_vector(
                        parameter_i)
    
    def get_parameter_vector_name_3d(self):
        """Get name of all parameters (3D)
        """
        parameter_name = []
        for time_series_lat in self.time_series_array:
            parameter_name.append([])
            for time_series_i in time_series_lat:
                parameter_name[-1].append(
                    time_series_i.get_parameter_vector_name())
        parameter_name = np.asarray(parameter_name)
        return parameter_name
    
    def get_parameter_vector_name(self):
        """Get name of all parameters (vector)
        """
        parameter_name_3d = self.get_parameter_vector_name_3d()
        parameter_name_array = []
        for i in range(self.n_parameter):
            parameter_name_array.append(
                parameter_name_3d[np.logical_not(self.mask), i].flatten())
        return np.asarray(parameter_name_array).flatten()
    
    def update_all_cp_parameters(self):
        """Update all compound Poisson parameters in all time series
        """
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    time_series.update_all_cp_parameters()
    
    def set_parameter_from_prior(self):
        """Set parameter from a sample from the prior
        """
        self.gp_target.precision = self.gp_target.simulate_from_prior(self.rng)
        self.precision_target.precision = (
            self.precision_target.simulate_from_prior(self.rng))
        self.update_reg_gp()
        self.update_reg_precision()
        parameter = self.parameter_target.simulate_from_prior(self.rng)
        self.set_parameter_vector(parameter)
    
    def update_reg_gp(self):
        """Propagate the GP precision to the parameter prior covariance
        """
        cov_chol = self.gp_target.get_cov_chol()
        self.parameter_target.prior_cov_chol = cov_chol
    
    def update_reg_precision(self):
        """Propagate the precision to the parameter prior covariance
        """
        precision = self.precision_target.precision
        self.parameter_target.prior_scale_parameter = 1 / math.sqrt(
            precision[0])
        self.parameter_target.prior_scale_arma = 1 / math.sqrt(
            precision[1])
    
    def get_log_likelihood(self):
        """Return log likelihood
        """
        ln_l_array = []
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    ln_l = time_series.get_joint_log_likelihood()
                    ln_l_array.append(ln_l)
        return np.sum(ln_l_array)
    
    def forecast(self, data, n_simulation):
        forecast_array = []
        i = 0
        area_unmask = self.area_unmask
        n_total_parameter = self.n_total_parameter
        for lat_i in range(self.shape[0]):
            forecast_array.append([])
            for long_i in range(self.shape[1]):
                x_i = data.get_model_field(lat_i, long_i)
                is_mask = self.mask[lat_i, long_i]
                if self.mask[lat_i, long_i]:
                    forecast = None
                else:
                    time_series = self.time_series_array[lat_i][long_i]
                    parameter_mcmc = np.array(
                        self.parameter_mcmc.sample_array)
                    parameter_mcmc = parameter_mcmc[
                        :, range(i, n_total_parameter, area_unmask)]
                    time_series.parameter_mcmc = parameter_mcmc
                    forecast = time_series.forecast(x_i, n_simulation)
                    i += 1
                forecast_array[-1].append(forecast)
        return forecast_array
