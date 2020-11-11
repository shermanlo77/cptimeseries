import math
import os
from os import path
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from numpy import random

import compound_poisson
from compound_poisson import forecast
from compound_poisson import mcmc
from compound_poisson import multiprocess
from compound_poisson import time_series_mcmc
from compound_poisson.mcmc import target_downscale

class Downscale(object):
    """Collection of multiple TimeSeries objects

    Fit a compound Poisson time series on multiple locations in 2d space.
        Parameters have a Gaussian process (GP) prior with gamma hyper
        parameters

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
        parameter_gp_target: TargetGp object
        parameter_mcmc: Mcmc object wrapping around parameter_target
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

    def __init__(self, data, n_arma=(0,0), is_test=False):
        #is_test argument used by DownscaleForGp (see forecast/downscale.py)
        #data is compatible with era5 as well by assuming data.model_field is
            #None
        self.n_arma = n_arma
        self.time_series_array = []
        self.time_array = data.time_array
        self.model_field_units = None
        self.n_model_field = None
        self.mask = data.mask
        self.parameter_mask_vector = []
        self.n_parameter = None
        self.n_total_parameter = None
        self.topography = data.topography
        self.topography_normalise = None
        self.shape = self.mask.shape
        self.area = self.shape[0] * self.shape[1]
        self.area_unmask = np.sum(np.logical_not(self.mask))
        self.seed_seq = None
        self.rng = None
        self.parameter_target = None
        self.parameter_gp_target = None
        self.parameter_mcmc = None
        self.parameter_gp_mcmc = None
        self.z_mcmc = None
        self.n_sample = 10000
        self.gibbs_weight = [0.003*len(self), 1, 0.2]
        self.burn_in = 0
        self.model_field_shift = []
        self.model_field_scale = []
        self.square_error = np.zeros((self.area_unmask, self.area_unmask))
        self.pool = None
        self.memmap_dir = ""
        self.forecaster = None

        if not data.model_field is None:
            self.model_field_units = data.model_field_units
            self.n_model_field = len(data.model_field)
            self.topography_normalise = data.topography_normalise

            #get the square error matrix used for GP
            unmask = np.logical_not(self.mask).flatten()
            for topo_i in self.topography_normalise.values():
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

        #instantiate time series for every point in space
        #unmasked points have rain, provide it to the constructor to
            #TimeSeries
        #masked points do not have rain, cannot provide it
        time_series_array = self.time_series_array
        TimeSeries = self.get_time_series_class()
        for lat_i in range(self.shape[0]):
            time_series_array.append([])
            for long_i in range(self.shape[1]):

                if data.model_field is None:
                    time_series = TimeSeries()
                else:

                    x_i, rain_i = data.get_data(lat_i, long_i)
                    is_mask = self.mask[lat_i, long_i]
                    if is_mask or is_test:
                        #provide no rain if this space is masked
                        #or the data provided is the test set
                        time_series = TimeSeries(x_i,
                                                 poisson_rate_n_arma=n_arma,
                                                 gamma_mean_n_arma=n_arma)
                    else:
                        #provide rain
                        time_series = TimeSeries(
                            x_i, rain_i.data, n_arma, n_arma)

                    for i in range(time_series.n_parameter):
                        self.parameter_mask_vector.append(is_mask)
                    self.n_parameter = time_series.n_parameter
                #provide information to time_series
                time_series.id = [lat_i, long_i]
                time_series.x_shift = self.model_field_shift
                time_series.x_scale = self.model_field_scale
                time_series.time_array = self.time_array
                time_series.memmap_dir = self.memmap_dir
                time_series_array[lat_i].append(time_series)

        if not data.model_field is None:
            #set other member variables
            self.set_seed_seq(random.SeedSequence())
            self.set_time_series_rng()
            self.parameter_mask_vector = np.asarray(self.parameter_mask_vector)
            self.n_total_parameter = self.area_unmask * self.n_parameter
            self.parameter_target = target_downscale.TargetParameter(self)
            self.parameter_gp_target = target_downscale.TargetGp(self)

    def get_time_series_class(self):
        return TimeSeriesDownscale

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

    def instantiate_mcmc(self):
        """Instantiate MCMC objects
        """
        self.parameter_mcmc = mcmc.Elliptical(
            self.parameter_target, self.rng, self.n_sample, self.memmap_dir)
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
            self.parameter_gp_mcmc,
        ]
        return mcmc_array

    def initalise_z(self):
        """Initalise z for all time series
        """
        #all time series initalise z, needed so that the likelihood can be
            #evaluated, eg y=0 if and only if x=0
        method = time_series_mcmc.static_initalise_z
        time_series_array = self.pool.map(
            method, self.generate_unmask_time_series())
        self.replace_unmask_time_series(time_series_array)

    def simulate_i(self, i):
        """Simulate a point in time for all time series
        """
        for time_series in self.generate_all_time_series():
            time_series.simulate_i(i)

    def simulate(self):
        """Simulate the entire time series for all time series
        """
        for time_series in self.generate_all_time_series():
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
        function = compound_poisson.time_series.static_update_all_cp_parameters
        time_series_array = self.pool.map(
            function, self.generate_unmask_time_series())
        self.replace_unmask_time_series(time_series_array)

    def get_log_likelihood(self):
        """Return log likelihood
        """
        method = (compound_poisson.time_series.TimeSeries
            .get_joint_log_likelihood)
        ln_l_array = self.pool.map(method, self.generate_unmask_time_series())
        return np.sum(ln_l_array)

    def forecast(self, data, n_simulation, pool=None):
        """Do forecast

        Args:
            data: test data
            n_simulation: number of Monte Carlo simulations
            pool: optional, a pool object to do parallel tasks

        Return:
            nested array of Forecast objects, shape corresponding to fine grid
        """
        if pool is None:
            pool = multiprocess.Serial()
        self.pool = pool
        self.read_memmap()
        self.scatter_mcmc_sample()

        if self.forecaster is None:
            self.forecaster = self.instantiate_forecaster()
            self.forecaster.start_forecast(n_simulation, data)
        else:
            self.forecaster.resume_forecast(n_simulation)

        self.del_memmap()

    def instantiate_forecaster(self):
        return forecast.downscale.Forecaster(self, self.memmap_dir)

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

        chain = np.asarray(self.parameter_gp_mcmc[:])
        for i, key in enumerate(self.parameter_gp_target.state):
            chain_i = chain[:, i]
            plt.plot(chain_i)
            plt.xlabel("sample number")
            plt.ylabel(key)
            plt.savefig(path.join(directory, key + "_downscale.pdf"))
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
        #note to developer: memory problem when parallelising
            #only for Cardiff for now (19, 22)
        message_array = []
        for i_space, time_series in enumerate(
            self.generate_unmask_time_series()):
            message = PlotMcmcMessage(
                self, chain, time_series, i_space, location_directory)
            message_array.append(message)
        pool.map(PlotMcmcMessage.print, message_array)

        self.del_memmap()

    def get_random_position_index(self):
        """Return array of random n_plot numbers, choose from
            range(self.area_unmask) without replacement

        Used to select random positions and plot their chains
        """
        n_plot = np.amin((6, self.area_unmask))
        rng = random.RandomState(np.uint32(4020967302))
        return np.sort(rng.choice(self.area_unmask, n_plot, replace=False))

    def generate_all_time_series(self):
        """Generate all time series in self.time_series_array
        """
        for time_series_i in self.time_series_array:
            for time_series in time_series_i:
                yield time_series

    def generate_unmask_time_series(self):
        """Generate all unmasked (on land) time series
        """
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    yield self.time_series_array[lat_i][long_i]

    def replace_single_time_series(self, time_series):
        """Replace a single time series in time_series_array, locataion is
            identified using time_series.id
        """
        lat_i = time_series.id[0]
        long_i = time_series.id[1]
        self.time_series_array[lat_i][long_i] = time_series

    def replace_unmask_time_series(self, time_series):
        """Replace all unmaksed time series with provided time series array

        Needed as parallel methods will do a deep copy of time_series objects
        """
        i = 0
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    self.time_series_array[lat_i][long_i] = time_series[i]
                    i += 1

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

    def scatter_mcmc_sample(self):
        self.scatter_z_mcmc_sample()
        self.scatter_parameter_mcmc_sample()

    def set_burn_in(self, burn_in):
        self.burn_in = burn_in
        for time_series in self.generate_unmask_time_series():
            time_series.burn_in = burn_in

    def set_memmap_dir(self, memmap_dir):
        """Set the location to save mcmc sample onto disk

        Modifies all mcmc objects to save the mcmc sample onto a specified
            location
        """
        self.memmap_dir = memmap_dir
        for time_series in self.generate_all_time_series():
            time_series.memmap_dir = memmap_dir

    def read_memmap(self):
        """Set memmap objects to read from file

        Required when loading saved object from joblib
        """
        for mcmc in self.get_mcmc_array():
            mcmc.read_memmap()

    def del_memmap(self):
        for mcmc in self.get_mcmc_array():
            mcmc.del_memmap()

    def delete_old_memmap(self):
        for mcmc in self.get_mcmc_array():
            mcmc.delete_old_memmap()

    def discard_sample(self, n_keep):
        """Discard initial mcmc samples to save hard disk space

        For each mcmc, make a new memmap and store the last n_keep mcmc samples.
            For saftey reasons, you will have to delete the old memmap file
            yourself.

        Args:
            n_keep: number of mcmc samples to keep (from the end of the chain)
        """
        for mcmc in self.get_mcmc_array():
            mcmc.discard_sample(n_keep)
        self.n_sample = n_keep

    def set_seed_seq(self, seed_sequence):
        self.seed_seq = seed_sequence
        self.rng = self.spawn_rng()

    def set_time_series_rng(self):
        """Set rng for all time series
        """
        for time_series in self.generate_all_time_series():
            time_series.set_rng(self.seed_seq)

    def set_rng(self, seed_sequence):
        """Set rng for all objects which uses rng

        To be override by subclasses
        """
        self.set_seed_seq(seed_sequence)
        self.set_time_series_rng()

    def spawn_rng(self, n=1):
        """Return array of substream rng

        Return array of independent random number generators by spawning from
            the seed sequence

        Return:
            array of numpy.random.RandomState objects if n > 1, or just a single
                object if n == 1
        """
        seed_spawn = self.seed_seq.spawn(n)
        rng_array = []
        for s in seed_spawn:
            rng_s = random.RandomState(random.MT19937(s))
            rng_array.append(rng_s)
        if len(rng_array) == 1:
            rng_array = rng_array[0]
        return rng_array

    def __len__(self):
        return len(self.time_array)

    def __getstate__(self):
        #required as multiprocessing cannot pickle multiprocessing.Pool
        self_dict = self.__dict__.copy()
        self_dict['pool'] = None
        return self_dict

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

class ForecastMessage(object):
    #message to pass, see Downscale.forecast

    def __init__(self, time_series, model_field, n_simulation):
        self.time_series = time_series
        self.model_field = model_field
        self.n_simulation = n_simulation

    def forecast(self):
        return self.time_series.forecast(self.model_field, self.n_simulation)

class PlotMcmcMessage(object):

    def __init__(self, downscale, chain, time_series, i_space, directory):
        self.time_series = time_series
        self.sub_directory = path.join(directory, str(time_series.id))
        if not path.isdir(self.sub_directory):
            os.mkdir(self.sub_directory)

    def print(self):
        self.time_series.print_mcmc(self.sub_directory)
