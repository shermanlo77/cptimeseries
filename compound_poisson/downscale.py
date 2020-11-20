import math
import os
from os import path

import matplotlib.pyplot as plt
import numpy as np

from compound_poisson import forecast
from compound_poisson import mcmc
from compound_poisson import multiprocess
from compound_poisson import multiseries
from compound_poisson import time_series_mcmc
from compound_poisson.mcmc import target_downscale

class Downscale(multiseries.MultiSeries):
    """Collection of multiple TimeSeries objects

    Fit a compound Poisson time series on multiple locations in 2d space.
        Parameters have a Gaussian process (GP) prior, each location has a
        log precision parameter with iid Normal prior. The GP precision
        has an inverse Gamma prior.

    Attributes:
        n_arma: 2-tuple, containing number of AR and MA terms
        time_series_array: 2d array containing TimeSeries objects, correspond to
            the fine grid
        time_array: array containing time stamp for each time step
        model_field_units: dictionary containing units for each model field,
            keys are strings describing the model field
        n_model_field: number of model fields
        mask: 2d boolean, True if on water, therefore masked
        parameter_mask_vector: mask as a vector
        n_parameter: number of parameters for one location
        n_total_parameter: n_parameter times number of unmasked time series
        topography: dictonary of topography information
        topography_normalise: dictonary of topography information normalised,
            mean 0, std 1
        shape: 2-tuple, shape of the space
        area: area of the space
        area_unmask: area of the unmasked space (number of points on fine grid)
        seed_seq: numpy.random.SeedSequence object
        rng: numpy.random.RandomState object
        parameter_target: TargetParameter object
        parameter_log_precision_target: TargetLogPrecisionGp object
        parameter_gp_target: TargetGp object
        parameter_mcmc: Mcmc object wrapping around parameter_target
        parameter_log_precision_mcmc: Mcmc object wrapping around
            parameter_log_precision_target
        parameter_gp_mcmc: Mcmc object wrapping around parameter_gp_target
        z_mcmc: array of all ZMcmc objects (owned by each unmasked time series)
        n_sample: number of mcmc samples
        gibbs_weight: probability of sampling each mcmc in self.get_mcmc_array()
        burn_in: number of initial mcmc samples to discard when forecasting
        model_field_shift: mean of model field, vector, entry for each model
            field
        model_field_scale: std of model field, vector, entry of reach model
            field
        square_error: matrix (area_unmask x area_unmask) containing square error
            of topography between each point in space
        pool: object for parallel programming
        memmap_dir: location to store mcmc samples and forecasts
    """

    def __init__(self, data, n_arma=(0,0)):
        #data is compatible with era5 as well by assuming data.model_field is
            #None
        super().__init__(data, n_arma)
        self.topography_normalise = None
        self.parameter_target = None
        self.parameter_log_precision_target = None
        self.parameter_gp_target = None
        self.parameter_mcmc = None
        self.parameter_log_precision_mcmc = None
        self.parameter_gp_mcmc = None
        self.z_mcmc = None
        self.n_sample = 10000
        self.gibbs_weight = [0.003*len(self), 1, 0.2, 0.2]
        self.model_field_shift = []
        self.model_field_scale = []
        self.square_error = np.zeros((self.area_unmask, self.area_unmask))
        self.pool = None

        if not data.model_field is None:
            self.model_field_units = data.model_field_units
            self.n_model_field = len(data.model_field)
            self.topography_normalise = data.topography_normalise

            #get the square error matrix used for GP
            #only use longitude and latitude
            unmask = np.logical_not(self.mask).flatten()
            for topo_i_key in ["longitude", "latitude"]:
                topo_i = self.topography_normalise[topo_i_key]
                topo_i = topo_i.flatten()
                topo_i = topo_i[unmask]
                for i in range(self.area_unmask):
                    for j in range(i+1, self.area_unmask):
                        self.square_error[i,j] += math.pow(
                            topo_i[i] - topo_i[j], 2)
                        self.square_error[j,i] = self.square_error[i,j]

            #get normalising info for model fields using mean and standard
                #deviation over all space and time
            for model_field in data.model_field.values():
                self.model_field_shift.append(np.mean(model_field))
                self.model_field_scale.append(np.std(model_field, ddof=1))
            self.model_field_shift = np.asarray(self.model_field_shift)
            self.model_field_scale = np.asarray(self.model_field_scale)

        for time_series_array_i in self.time_series_array:
            for time_series_i in time_series_array_i:
                time_series_i.x_shift = self.model_field_shift
                time_series_i.x_scale = self.model_field_scale
                time_series_i.memmap_dir = self.memmap_dir

        if not data.model_field is None:
            #set target
            self.parameter_target = target_downscale.TargetParameter(self)
            self.parameter_log_precision_target = (
                target_downscale.TargetLogPrecision(self))
            self.parameter_gp_target = target_downscale.TargetGp(self)

    #override
    def get_time_series_class(self):
        return TimeSeriesDownscale

    #override
    def fit(self, pool=None):
        """Fit using Gibbs sampling

        Args:
            pool: optional, a pool object to do parallel tasks
        """
        if pool is None:
            pool = multiprocess.Serial()
        self.pool = pool
        self.initalise_z()
        self.instantiate_mcmc()
        mcmc_array = self.get_mcmc_array()
        mcmc.do_gibbs_sampling(mcmc_array, self.n_sample, self.rng,
                self.gibbs_weight)
        self.scatter_mcmc_sample()
        self.del_memmap()
        self.pool = None

    #override
    def resume_fitting(self, n_sample, pool=None):
        """Run more MCMC samples

        Args:
            n_sample: new number of mcmc samples
        """
        if pool is None:
            pool = multiprocess.Serial()
        self.pool = pool
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
            self.scatter_mcmc_sample()
        self.del_memmap()
        self.pool = None

    #override
    def instantiate_mcmc(self):
        """Instantiate MCMC objects
        """
        self.parameter_mcmc = mcmc.Elliptical(
            self.parameter_target, self.rng, self.n_sample, self.memmap_dir)
        self.parameter_log_precision_mcmc = mcmc.Elliptical(
            self.parameter_log_precision_target, self.rng, self.n_sample,
            self.memmap_dir)
        self.parameter_gp_mcmc = mcmc.Rwmh(
            self.parameter_gp_target, self.rng, self.n_sample, self.memmap_dir)
        #all time series objects instantiate mcmc objects to store the z chain
        for time_series in self.generate_unmask_time_series():
            time_series.n_sample = self.n_sample
            time_series.instantiate_mcmc()
        self.z_mcmc = mcmc.ZMcmcArray(self)
        self.parameter_gp_target.save_cov_chol()

    def get_mcmc_array(self):
        """Return array of all mcmc objects
        """
        mcmc_array = [
            self.z_mcmc,
            self.parameter_mcmc,
            self.parameter_log_precision_mcmc,
            self.parameter_gp_mcmc,
        ]
        return mcmc_array

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

    #override
    def print_mcmc(self, directory, pool):
        """Print the mcmc chains
        """
        directory = path.join(directory, "chain")
        if not path.isdir(directory):
            os.mkdir(directory)
        location_directory = path.join(directory, "locations")
        if not path.isdir(location_directory):
            os.mkdir(location_directory)
        self.read_memmap()
        self.scatter_mcmc_sample()
        position_index_array = self.get_random_position_index()

        #pick random locations and plot their mcmc chains
        chain = np.asarray(self.parameter_mcmc[:])
        area_unmask = self.area_unmask
        parameter_name = (
            self.time_series_array[0][0].get_parameter_vector_name())
        for i in range(self.n_parameter):
            chain_i = []
            for position_index in position_index_array:
                chain_i.append(chain[:, i*area_unmask + position_index])
            plt.plot(np.asarray(chain_i).T)
            plt.xlabel("sample number")
            plt.ylabel(parameter_name[i])
            plt.savefig(path.join(directory, "parameter_" + str(i) + ".pdf"))
            plt.close()

        chain = np.asarray(self.parameter_log_precision_mcmc[:])
        area_unmask = self.area_unmask
        parameter_name = list(self.parameter_log_precision_target.prior.keys())
        for i, parameter_name_i in enumerate(parameter_name):
            chain_i = []
            for position_index in position_index_array:
                chain_i.append(chain[:, i*area_unmask + position_index])
            plt.plot(np.asarray(chain_i).T)
            plt.xlabel("sample number")
            plt.ylabel(parameter_name_i)
            plt.savefig(path.join(directory, parameter_name_i + ".pdf"))
            plt.close()

        plt.figure()
        plt.plot(self.parameter_gp_mcmc[:])
        plt.xlabel("sample number")
        plt.ylabel("gp_precision")
        plt.savefig(path.join(directory, "gp_precision.pdf"))
        plt.close()

        chain = []
        for i, time_series in enumerate(self.generate_unmask_time_series()):
            if i in position_index_array:
                time_series.read_memmap()
                chain.append(np.mean(time_series.z_mcmc[:], 1))
                time_series.del_memmap()
        plt.plot(np.transpose(np.asarray(chain)))
        plt.xlabel("sample number")
        plt.ylabel("mean z")
        plt.savefig(path.join(directory, "z.pdf"))
        plt.close()

        #plot each chain for each location
        message_array = []
        for i_space, time_series in enumerate(
            self.generate_unmask_time_series()):
            message = multiseries.PlotMcmcMessage(
                self, chain, time_series, i_space, location_directory)
            message_array.append(message)
        pool.map(multiseries.PlotMcmcMessage.print, message_array)

        self.del_memmap()

    #override
    def scatter_mcmc_sample(self):
        self.scatter_z_mcmc_sample()
        self.scatter_parameter_mcmc_sample()

    def scatter_z_mcmc_sample(self):
        z_mcmc = self.z_mcmc
        for i, time_series in enumerate(self.generate_unmask_time_series()):
            z_mcmc_i = time_series.z_mcmc
            z_mcmc_i.set_memmap_slice(z_mcmc.n_sample,
                                      z_mcmc.n_dim,
                                      z_mcmc.memmap_path,
                                      slice(i*len(self), (i+1)*len(self)))

    def scatter_parameter_mcmc_sample(self):
        parameter_mcmc = self.parameter_mcmc
        for i, time_series in enumerate(self.generate_unmask_time_series()):
            parameter_mcmc_i = mcmc.Mcmc(parameter_mcmc.dtype)
            slice_index = slice(i, self.n_total_parameter, self.area_unmask)
            parameter_mcmc_i.set_memmap_slice(parameter_mcmc.n_sample,
                                              parameter_mcmc.n_dim,
                                              parameter_mcmc.memmap_path,
                                              slice_index)
            time_series.parameter_mcmc = parameter_mcmc_i

    #override
    def read_memmap(self):
        """Set memmap objects to read from file

        Required when loading saved object from joblib
        """
        for mcmc in self.get_mcmc_array():
            mcmc.read_memmap()

    #override
    def del_memmap(self):
        for mcmc in self.get_mcmc_array():
            mcmc.del_memmap()

    #override
    def delete_old_memmap(self):
        for mcmc in self.get_mcmc_array():
            mcmc.delete_old_memmap()

class TimeSeriesDownscale(time_series_mcmc.TimeSeriesSlice):
    """Modify TimeSeriesSlice to only sample z
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
        self.z_mcmc = None

    def instantiate_mcmc(self):
        """Instantiate all MCMC objects

        Override
        Only instantiate slice sampling for z
        """
        self.parameter_mcmc = None
        self.z_mcmc = mcmc.ZSlice(self.z_target, self.rng)

    def forecast(self, x, n_simulation, memmap_path, memmap_shape, i_space):
        #override to include MCMC samples
        self.read_memmap()
        if self.forecaster is None:
            self.forecaster = forecast.downscale.TimeSeriesForecaster(
                self, memmap_path, i_space)
            self.forecaster.start_forecast(n_simulation, x, memmap_shape)
        else:
            self.forecaster.memmap_path = memmap_path
            self.forecaster.resume_forecast(n_simulation, memmap_shape)
        self.del_memmap()

    def print_chain_property(self, directory):
        """Override: There are no chain properties to plot, eg acceptance rate.
            For chain properties, look for mcmc objects in the Downscale object
            which should own instances of this class
        """
        pass

class DownscaleDeepGp(Downscale):
    """Extension of Downscale by putting a GP prior on the log precision
        parameters. The GP precision has an inverse Gamma prior.

    Attributes:
        parameter_log_precision_target: TargetLogPrecisionGp object
        parameter_log_precision_gp_target: TargetDeepGp object
        parameter_log_precision_gp_mcmc: Mcmc object wrapping around
            parameter_log_precision_gp_target
    """

    def __init__(self, data, n_arma=(0,0)):
        self.parameter_log_precision_gp_target = None
        self.parameter_log_precision_gp_mcmc = None
        super().__init__(data, n_arma)
        if not data.model_field is None:
            self.parameter_log_precision_target = (
                target_downscale.TargetLogPrecisionGp(self))
            self.parameter_log_precision_gp_target = (
                target_downscale.TargetDeepGp(self))
        self.gibbs_weight = [0.003*len(self), 1, 0.2, 0.2, 0.2]

    #override
    def instantiate_mcmc(self):
        super().instantiate_mcmc()
        self.parameter_log_precision_gp_mcmc = mcmc.Rwmh(
            self.parameter_log_precision_gp_target, self.rng, self.n_sample,
            self.memmap_dir)
        self.parameter_log_precision_gp_target.save_cov_chol()

    #override
    def get_mcmc_array(self):
        mcmc_array = super().get_mcmc_array()
        mcmc_array.append(self.parameter_log_precision_gp_mcmc)
        return mcmc_array

    #override
    def print_mcmc(self, directory, pool):
        super().print_mcmc(directory, pool)

        self.read_memmap()

        directory = path.join(directory, "chain")
        plt.figure()
        plt.plot(self.parameter_log_precision_gp_mcmc[:])
        plt.xlabel("sample number")
        plt.ylabel("log_precision_gp_precision")
        plt.savefig(path.join(directory, "log_precision_gp_precision.pdf"))
        plt.close()
        self.del_memmap()
