import math

import numpy as np
from numpy import random

import compound_poisson
from compound_poisson import mcmc
from compound_poisson.mcmc import target_downscale
from compound_poisson.mcmc import target_model_field

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
        parameter_precision_mcmc: TargetPrecision object
        parameter_gp_target: TargetGp object
        parameter_mcmc: Mcmc object wrapping around parameter_target
        parameter_precision_mcmc: Mcmc object wrapping around
            parameter_precision_target
        gp_mcmc: Mcmc object wrapping around parameter_gp_target
        n_sample: number of MCMC samples
        model_field_shift: mean of model field, vector, entry for each model
            field
        model_field_scale: std of model field, vector, entry of reach model
            field
    """

    def __init__(self, data, n_arma=(0,0)):
        self.n_arma = n_arma
        self.time_series_array = []
        self.time_array = data.time_array
        self.model_field_units = data.model_field_units
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
        self.parameter_gp_target = None
        self.parameter_mcmc = None
        self.parameter_gp_mcmc = None
        self.n_sample = 10000
        self.model_field_shift = []
        self.model_field_scale = []
        self.n_model_field = len(data.model_field)
        self.square_error = np.zeros((self.area_unmask, self.area_unmask))

        #get the square error matrix used for GP
        unmask = np.logical_not(self.mask).flatten()
        for topo_i in self.topography_normalise.values():
            topo_i = topo_i.flatten()
            topo_i = topo_i[unmask]
            for i in range(self.area_unmask):
                for j in range(i+1, self.area_unmask):
                    self.square_error[i,j] += math.pow(topo_i[i] - topo_i[j], 2)
                    self.square_error[j,i] = self.square_error[i,j]

        #get normalising info for model fields using mean and standard deviation
            #over all space and time
        for model_field in data.model_field.values():
            self.model_field_shift.append(np.mean(model_field))
            self.model_field_scale.append(np.std(model_field, ddof=1))
        self.model_field_shift = np.asarray(self.model_field_shift)
        self.model_field_scale = np.asarray(self.model_field_scale)

        #instantiate time series for every point in space
        #unmasked points have rain, provide it to the constructor to TimeSeries
        #masked points do not have rain, cannot provide it
        time_series_array = self.time_series_array
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
                time_series.x_shift = self.model_field_shift
                time_series.x_scale = self.model_field_scale
                time_series.time_array = self.time_array
                time_series_array[lat_i].append(time_series)
                for i in range(time_series.n_parameter):
                    self.parameter_mask_vector.append(is_mask)
                self.n_parameter = time_series.n_parameter

        self.parameter_mask_vector = np.asarray(self.parameter_mask_vector)
        self.n_total_parameter = self.area_unmask * self.n_parameter
        self.parameter_target = target_downscale.TargetParameter(self)
        self.parameter_gp_target = target_downscale.TargetGp(self)

    def fit(self):
        """Fit using Gibbs sampling
        """
        self.initalise_z()
        self.instantiate_mcmc()
        self.update_parameter_gp()

        #initial value is a sample
        self.parameter_mcmc.add_to_sample()
        self.parameter_gp_mcmc.add_to_sample()
        self.z_mcmc_add_to_sample()

        #Gibbs sampling
        for i in range(self.n_sample):
            print("Sample",i)
            #select random component
            rand = self.rng.rand()
            if rand < 1/3:
                self.z_mcmc_step()
                self.parameter_mcmc.add_to_sample()
                self.parameter_gp_mcmc.add_to_sample()
            elif rand < 2/3:
                self.parameter_mcmc.step()
                self.parameter_gp_mcmc.add_to_sample()
                self.z_mcmc_add_to_sample()
            else:
                self.parameter_gp_mcmc.step()
                self.update_parameter_gp()
                self.parameter_mcmc.add_to_sample()
                self.z_mcmc_add_to_sample()

    def z_mcmc_add_to_sample(self):
        """Duplicate z_array to the z chain

        For all time_series, add z_array to the sample_array
        """
        for time_series in self.generate_unmask_time_series():
            time_series.z_mcmc.add_to_sample()

    def z_mcmc_step(self):
        """All time_series sample z
        """
        for time_series in self.generate_unmask_time_series():
            time_series.z_mcmc.step()

    def instantiate_mcmc(self):
        """Instantiate MCMC objects
        """
        self.parameter_mcmc = mcmc.Elliptical(self.parameter_target, self.rng)
        self.parameter_gp_mcmc = mcmc.Rwmh(self.parameter_gp_target, self.rng)
        self.parameter_gp_mcmc.proposal_covariance_small = 1e-4
        #all time series objects instantiate mcmc objects to store the z chain
        for time_series in self.generate_unmask_time_series():
            time_series.instantiate_mcmc()

    def initalise_z(self):
        """Initalise z for all time series
        """
        #all time series initalise z, needed so that the likelihood can be
            #evaluated, eg y=0 if and only if x=0
        for time_series in self.generate_unmask_time_series():
            time_series.initalise_z()
        self.update_all_cp_parameters()

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
        for time_series in self.generate_unmask_time_series():
            time_series.update_all_cp_parameters()

    def update_parameter_gp(self):
        """Propagate the GP precision to the parameter prior covariance
        """
        self.parameter_gp_target.save_cov_chol()

    def get_log_likelihood(self):
        """Return log likelihood
        """
        ln_l_array = []
        for time_series in self.generate_unmask_time_series():
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

    def generate_unmask_time_series(self):
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    yield self.time_series_array[lat_i][long_i]

    def __len__(self):
        return len(self.time_array)

class DownscaleDual(Downscale):

    def __init__(self, data, n_arma=(0,0)):
        super().__init__(data, n_arma)
        self.model_field_coarse = data.model_field
        self.topography_coarse = data.topography_coarse
        self.topography_coarse_normalise = data.topography_coarse_normalise
        self.n_coarse = None
        self.model_field_target = None
        self.model_field_mcmc = None
        self.model_field_gp_target = None

        for model_field in self.model_field_coarse.values():
            self.n_coarse = model_field[0].size
            break

        self.model_field_gp_target = target_model_field.TargetGp(self)
        self.model_field_target = target_model_field.TargetModelFieldArray(self)

    def get_model_field(self, time_step):
        """Return model field for all unmasked time_series

        Return:
            vector of length self.n_model_field * self.area_unmask, [0th model
                field for all unmasked, 1st model field for all unmasked, ...]
        """
        model_field = []
        for model_field_i in range(self.n_model_field):
            for time_series in self.generate_unmask_time_series():
                model_field.append(time_series.x[time_step, model_field_i])
        return np.asarray(model_field)

    def set_model_field(self, model_field_vector, time_step):
        """Set the model field for all unmasked time_series

        Args:
            vector of length self.n_model_field * self.area_unmask, [0th model
                field for all unmasked, 1st model field for all unmasked, ...]
        """
        i_counter = 0
        for model_field_i in range(self.n_model_field):
            for time_series in self.generate_unmask_time_series():
                time_series.x[time_step, model_field_i] = model_field_vector[
                    i_counter]
                i_counter += 1

    def get_model_field_log_pdf(self):
        """Return the log pdf of all model fields for all time steps
        """
        ln_pdf = []
        for model_field in self.model_field_target:
            ln_pdf.append(model_field.get_log_prior())
        return np.sum(ln_pdf)

    def update_model_field_gp(self):
        """Propagate the GP precision to the parameter prior covariance
        """
        self.model_field_gp_target.save_cov_chol()
        self.model_field_target.update_mean()

    def update_model_field_gp_i(self, time_step):
        self.model_field_gp_target.save_cov_chol()
        self.model_field_target[time_step].update_mean()
