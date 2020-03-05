import math

import numpy as np
from numpy.random import RandomState

from .mcmc import Elliptical
from .mcmc import Rwmh
from .mcmc import TargetDownscaleGp
from .mcmc import TargetDownscaleParameter
from .mcmc import TargetDownscalePrecision
from .TimeSeriesSlice import TimeSeriesSlice

class Downscale:
    
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
        self.rng = RandomState()
        self.parameter_target = None
        self.precision_target = None
        self.gp_target = None
        self.parameter_mcmc = None
        self.precision_mcmc = None
        self.gp_mcmc = None
        self.n_sample = 10000
        
        model_field = data.model_field
        rain = data.rain
        time_series_array = self.time_series_array
        
        for lat_i in range(self.shape[0]):
            
            time_series_array.append([])
            
            for long_i in range(self.shape[1]):
                
                x_i, rain_i = data.get_data(lat_i, long_i)
                is_mask = self.mask[lat_i, long_i]
                if is_mask:
                    time_series = TimeSeriesSlice(x_i,
                                                  poisson_rate_n_arma=n_arma,
                                                  gamma_mean_n_arma=n_arma)
                else:
                    time_series = TimeSeriesSlice(
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
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    time_series.z_mcmc.add_to_sample()
    
    def z_mcmc_step(self):
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    time_series.z_mcmc.step()
    
    def instantiate_mcmc(self):
        self.parameter_mcmc = Elliptical(self.parameter_target, self.rng)
        self.precision_mcmc = Rwmh(self.precision_target, self.rng)
        self.precision_mcmc.proposal_covariance_small = 1e-4
        self.gp_mcmc = Rwmh(self.gp_target, self.rng)
        self.gp_mcmc.proposal_covariance_small = 1e-4
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    time_series.instantiate_mcmc()
    
    def initalise_z(self):
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    time_series.initalise_z()
        self.update_all_cp_parameters()
    
    def set_target(self):
        self.parameter_target = TargetDownscaleParameter(self)
        self.precision_target = TargetDownscalePrecision(self.parameter_target)
        self.gp_target = TargetDownscaleGp(self)
    
    def set_rng(self, rng):
        self.rng = rng
        for time_series_i in self.time_series_array:
            for time_series in time_series_i:
                time_series.rng = rng
    
    def simulate_i(self, i):
        for time_series_i in self.time_series_array:
            for time_series in time_series_i:
                time_series.simulate_i(i)
    
    def simulate(self):
        for time_series_i in self.time_series_array:
            for time_series in time_series_i:
                time_series.simulate()
    
    def get_parameter_3d(self):
        parameter = []
        for time_series_lat in self.time_series_array:
            parameter.append([])
            for time_series_i in time_series_lat:
                parameter[-1].append(time_series_i.get_parameter_vector())
        parameter = np.asarray(parameter)
        return parameter
    
    def get_parameter_vector(self):
        parameter_3d = self.get_parameter_3d()
        parameter_vector = []
        for i in range(self.n_parameter):
            parameter_vector.append(
                parameter_3d[np.logical_not(self.mask), i].flatten())
        return np.asarray(parameter_vector).flatten()
    
    def set_parameter_vector(self, parameter_vector):
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
        parameter_name = []
        for time_series_lat in self.time_series_array:
            parameter_name.append([])
            for time_series_i in time_series_lat:
                parameter_name[-1].append(
                    time_series_i.get_parameter_vector_name())
        parameter_name = np.asarray(parameter_name)
        return parameter_name
    
    def get_parameter_vector_name(self):
        parameter_name_3d = self.get_parameter_vector_name_3d()
        parameter_name_array = []
        for i in range(self.n_parameter):
            parameter_name_array.append(
                parameter_name_3d[np.logical_not(self.mask), i].flatten())
        return np.asarray(parameter_name_array).flatten()
    
    def update_all_cp_parameters(self):
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    time_series.update_all_cp_parameters()
    
    def set_parameter_from_prior(self):
        self.gp_target.precision = self.gp_target.simulate_from_prior(self.rng)
        self.precision_target.precision = (
            self.precision_target.simulate_from_prior(self.rng))
        self.update_reg_gp()
        self.update_reg_precision()
        parameter = self.parameter_target.simulate_from_prior(self.rng)
        self.set_parameter_vector(parameter)
    
    def update_reg_gp(self):
        cov_chol = self.gp_target.get_cov_chol()
        self.parameter_target.prior_cov_chol = cov_chol
    
    def update_reg_precision(self):
        precision = self.precision_target.precision
        self.parameter_target.prior_scale_parameter = 1 / math.sqrt(
            precision[0])
        self.parameter_target.prior_scale_arma = 1 / math.sqrt(
            precision[1])
    
    def get_log_likelihood(self):
        ln_l_array = []
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    time_series = self.time_series_array[lat_i][long_i]
                    ln_l = time_series.get_joint_log_likelihood()
                    ln_l_array.append(ln_l)
        return np.sum(ln_l_array)
