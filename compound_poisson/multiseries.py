"""
TODO: Suggestion, rewrite the methods get_parameter_3d(), get_parameter_vector()
    set_parameter_vector() to make use of newer code
"""

import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
from numpy import random

import compound_poisson
from compound_poisson import forecast
from compound_poisson import mcmc
from compound_poisson import multiprocess
from compound_poisson import time_series_mcmc

class MultiSeries(object):
    """Collection of multiple TimeSeries objects

    Fit a compound Poisson time series on multiple locations in 2d space.
        Fitting is all done in parallel and independently.

    Attributes:
        n_arma: 2-tuple, containing number of AR and MA terms
        time_series_array: 2d array containing TimeSeries objects, correspond to
            the fine grid
        time_array: array containing time stamp for each time step
        model_field_units: dictionary containing units for each model field,
            keys are strings describing the model field
        n_model_field: number of model fields
        mask: 2d boolean, True if on water, therefore masked
        parameter_mask_vector: mask, of all parameters, as a vector.
        n_parameter: number of parameters for one location
        n_total_parameter: n_parameter times number of unmasked time series
        topography: dictionary of topography information
        shape: 2-tuple, shape of the space
        area: area of the space
        area_unmask: area of the unmasked space (number of points on fine grid)
        seed_seq: numpy.random.SeedSequence object
        rng: numpy.random.RandomState object
        n_sample: number of mcmc samples
        burn_in: number of initial mcmc samples to discard when forecasting
        pool: object for parallel programming
        memmap_dir: location to store mcmc samples and forecasts
        mcmc: MultiMcmc object, containing each locations' Mcmc
        model_field_shift: mean of model field, vector, entry for each model
            field
        model_field_scale: std of model field, vector, entry of reach model
            field
    """

    def __init__(self, data, n_arma=(0,0)):
        """
        Args:
            data: DataDualGrid object containing the training set
        """
        #note: data can have no model fields (eg ERA5)
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
        self.shape = self.mask.shape
        self.area = self.shape[0] * self.shape[1]
        self.area_unmask = np.sum(np.logical_not(self.mask))
        self.seed_seq = None
        self.rng = None
        self.n_sample = 10000
        self.burn_in = 0
        self.pool = None
        self.memmap_dir = ""
        self.forecaster = None
        self.mcmc = None
        self.model_field_shift = []
        self.model_field_scale = []

        #instantiate time series for every point in space
        #unmasked points have rain, provide it to the constructor to TimeSeries
        #masked points do not have rain, cannot provide it
        time_series_array = self.time_series_array
        TimeSeries = self.get_time_series_class()
        for lat_i in range(self.shape[0]):
            time_series_array.append([])
            for long_i in range(self.shape[1]):
                #TimeSeries object to be appended to time_series_array
                time_series = None
                #empty constructor for TimeSeries is non data
                if data.model_field is None:
                    time_series = TimeSeries()
                else:
                    x_i, rain_i = data.get_data(lat_i, long_i)
                    is_mask = self.mask[lat_i, long_i]
                    if is_mask:
                        #provide no rain if this space is masked
                        time_series = TimeSeries(x_i,
                                                 poisson_rate_n_arma=n_arma,
                                                 gamma_mean_n_arma=n_arma)
                    else:
                        #provide rain
                        time_series = TimeSeries(
                            x_i, rain_i.data, n_arma, n_arma)
                    #
                    for i in range(time_series.n_parameter):
                        self.parameter_mask_vector.append(is_mask)
                    self.n_parameter = time_series.n_parameter

                #provide information to time_series
                time_series.id = [lat_i, long_i]
                time_series.time_array = self.time_array
                time_series_array[lat_i].append(time_series)

        if not data.model_field is None:
            #set other member variables
            self.model_field_units = data.model_field_units
            self.n_model_field = len(data.model_field)
            self.set_seed_seq(random.SeedSequence())
            self.set_time_series_rng()
            self.parameter_mask_vector = np.asarray(self.parameter_mask_vector)
            self.n_total_parameter = self.area_unmask * self.n_parameter

            #get normalising info for model fields using mean and standard
                #deviation over all space and time
            #all locations share the same normalisation constants
            for model_field in data.model_field.values():
                self.model_field_shift.append(np.mean(model_field))
                self.model_field_scale.append(np.std(model_field, ddof=1))
            self.model_field_shift = np.asarray(self.model_field_shift)
            self.model_field_scale = np.asarray(self.model_field_scale)
            for time_series_array_i in self.time_series_array:
                for time_series_i in time_series_array_i:
                    time_series_i.x_shift = self.model_field_shift
                    time_series_i.x_scale = self.model_field_scale

    def get_time_series_class(self):
        """Return the class used for the array of time series
        """
        return TimeSeriesMultiSeries

    def fit(self, pool=None):
        """Fit using Gibbs sampling, does self.n_sample MCMC samples

        Args:
            pool: optional, a pool object to do parallel tasks
        """
        if pool is None:
            pool = multiprocess.Serial()
        self.pool = pool
        self.initalise_z()
        self.instantiate_mcmc()
        self.scatter_mcmc_sample()
        #parallel fit over locations
        message_array = []
        for time_series_i in self.generate_unmask_time_series():
            message_array.append(FitMessage(time_series_i))
        time_series_array = self.pool.map(FitMessage.fit, message_array)
        self.replace_unmask_time_series(time_series_array)

        self.del_memmap()

        self.pool = None

    def resume_fitting(self, n_sample, pool=None):
        """Run more MCMC samples

        Run more MCMC samples. Deep copy the MCMC memmaps and add new MCMC
            samples to the copied memmap. The old memmap file is deleted after
            a successful execuation of MCMC steps.

        Args:
            n_sample: new number of mcmc samples, must be higher than
                self.n_sample
            pool: optional, a pool object to do parallel tasks
        """
        if pool is None:
            pool = multiprocess.Serial()
        self.pool = pool
        if n_sample > self.n_sample:
            self.n_sample = n_sample
            self.mcmc.extend_memmap(n_sample)
            self.scatter_mcmc_sample()

            message_array = []
            for time_series_i in self.generate_unmask_time_series():
                message_array.append(
                    ResumeFittingMessage(time_series_i, n_sample))
            time_series_array = self.pool.map(
                ResumeFittingMessage.fit, message_array)
            self.replace_unmask_time_series(time_series_array)

            self.delete_old_memmap()
        self.del_memmap()
        self.pool = None

    def instantiate_mcmc(self):
        """Instantiate MCMC objects
        """
        #do not provide memmap so that when instantiating mcmc for each time
            #series, it does not create a memmap file
        for time_series_i in self.generate_unmask_time_series():
            time_series_i.instantiate_mcmc()
        self.mcmc = MultiMcmc(self)

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

    def get_parameter_3d(self):
        """Return the parameters from all time series (3D)

        Return a 3D array of all the parameters in all unmasked time series
            dim 0: for each latitude
            dim 1: for each longitude
            dim 2: for each parameter
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

    def forecast(self,
                 data,
                 n_simulation,
                 pool=None,
                 use_gp=False,
                 topo_key=["latitude", "longitude"]):
        """Do forecast, updates the member variable forecaster

        Args:
            data: test data
            n_simulation: number of Monte Carlo simulations
            pool: optional, a pool object to do parallel tasks
            use_gp: optional, True if to use post-sampling GP smoothing,
                defaults to False
            topo_key: optional, to be used if use_gp is True, array of
                topography keys to use as gp inputs
        """
        if pool is None:
            pool = multiprocess.Serial()
        self.pool = pool
        self.read_memmap()
        self.scatter_mcmc_sample()

        if self.forecaster is None:
            self.forecaster = self.instantiate_forecaster(use_gp, topo_key)
            self.forecaster.start_forecast(n_simulation, data)
        else:
            self.forecaster.resume_forecast(n_simulation)

        self.del_memmap()

    def instantiate_forecaster(self, use_gp=False, topo_key=None):
        """Instantiate a Forecaster object and return it

        Args:
            use_gp: optional, True if to use post-sampling GP smoothing,
                defaults to False
            topo_key: optional, to be used if use_gp is True, array of
                topography keys to use as gp inputs

        Return:
            instantiated Forecaster object
            topo_key: optional, to be used if use_gp is True, array of
                topography keys to use as gp inputs
        """
        if use_gp:
            forecaster = forecast.downscale.ForecasterGp(
                self, self.memmap_dir, topo_key)
        else:
            forecaster = forecast.downscale.Forecaster(self, self.memmap_dir)
        return forecaster

    def print_mcmc(self, directory, pool):
        """Print the mcmc chains

        Args:
            directory: where to save the figures
            pool: a pool object to do parallel tasks (at least can call map())
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
        for j, name in enumerate(self.mcmc.mcmc_name_array):
            chain = self.mcmc.mcmc_array[j].sample_array
            chain_dim = self.mcmc.n_dim_array[j]

            if "Parameter" in name:
                parameter_name = (
                    self.time_series_array[0][0].get_parameter_vector_name())
                for i in range(self.n_parameter):
                    chain_i = []
                    for position_index in position_index_array:
                        chain_i.append(chain[:, position_index*chain_dim + i])
                    plt.plot(np.asarray(chain_i).T)
                    plt.xlabel("sample number")
                    plt.ylabel(parameter_name[i])
                    plt.savefig(
                        path.join(directory, "parameter_" + str(i) + ".pdf"))
                    plt.close()

            elif "Precision" in name:
                for i in range(2):
                    chain_i = []
                    for position_index in position_index_array:
                        chain_i.append(chain[:, position_index*chain_dim + i])
                    plt.plot(np.asarray(chain_i).T)
                    plt.xlabel("sample number")
                    plt.ylabel("precision " + str(i))
                    plt.savefig(
                        path.join(directory, "precision_" + str(i) + ".pdf"))
                    plt.close()

            elif "TargetZ" in name:
                chain_i = []
                for i in position_index_array:
                    chain_i.append(
                        np.mean(chain[:, i*len(self):(i+1)*len(self)], 1))
                plt.plot(np.transpose(np.asarray(chain_i)))
                plt.xlabel("sample number")
                plt.ylabel("mean z")
                plt.savefig(path.join(directory, "z.pdf"))
                plt.close()

        #plot each chain for each location
        message_array = []
        for i_space, time_series in enumerate(
            self.generate_unmask_time_series()):
            message = PlotMcmcMessage(
                self, chain, time_series, i_space, location_directory)
            message_array.append(message)
        pool.map(PlotMcmcMessage.print, message_array)

        self.del_memmap()

    def get_random_position_index(self):
        """Return array of 6 andom numbers, choose from range(self.area_unmask)
            without replacement

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

        Needed as parallel methods will past a deep copy of time_series objects
            and return a reference to that deep copy
        """
        i = 0
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not self.mask[lat_i, long_i]:
                    self.time_series_array[lat_i][long_i] = time_series[i]
                    i += 1

    def scatter_mcmc_sample(self):
        """Scatter the MCMC samples to each location
        """
        self.mcmc.scatter_mcmc()

    def set_burn_in(self, burn_in):
        """Set the member variable burn-in

        Set the member variable burn-in for MultiSeries and each TimeSeries
            object
        """
        self.burn_in = burn_in
        for time_series in self.generate_unmask_time_series():
            time_series.burn_in = burn_in

    def set_memmap_dir(self, memmap_dir):
        """Set the location to save mcmc sample onto disk
        """
        self.memmap_dir = memmap_dir
        for time_series in self.generate_all_time_series():
            time_series.memmap_dir = memmap_dir

    def read_memmap(self):
        """Set memmap objects to read from file

        Required when loading saved object from joblib
        """
        self.mcmc.read_memmap()

    def del_memmap(self):
        """Close (del) references to memmap objects

        Used to prevent too many memmap files being opened
        """
        self.mcmc.del_memmap()

    def delete_old_memmap(self):
        """DANGEROUS: Actually deletes the old mcmc file
        """
        self.mcmc.delete_old_memmap()

    def discard_sample(self, n_keep):
        """For debugging: Discard initial mcmc samples to save hard disk space

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
        """Set the rng using the provided SeedSequence
        """
        self.seed_seq = seed_sequence
        self.rng = self.spawn_rng()

    def set_time_series_rng(self):
        """Set rng for all time series
        """
        for time_series in self.generate_all_time_series():
            time_series.set_rng(self.seed_seq)

    def set_rng(self, seed_sequence):
        """Set rng for all objects which uses rngs
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
        for seed in seed_spawn:
            rng_i = random.RandomState(random.MT19937(seed))
            rng_array.append(rng_i)
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

class TimeSeriesMultiSeries(time_series_mcmc.TimeSeriesHyperSlice):
    """Modify TimeSeriesSlice so that it shares a memmap with all other
        locations
    """

    def __init__(self,
                 x,
                 rainfall=None,
                 poisson_rate_n_arma=None,
                 gamma_mean_n_arma=None):
        super().__init__(x,
                         rainfall,
                         poisson_rate_n_arma,
                         gamma_mean_n_arma)
        #set these to None to ensure a memmap is not allocated at instantiation
            #allocation is done using MultiMcmc
        self.n_sample = None
        self.memmap_dir = None

    #override
    def fit(self):
        """Overridden to use the common memmap between locations

        Also modify the print statements so it prints the progress when a
            location has completed their fitting rather than for each gibbs
            step.
        The initalisation of z and Mcmc objects should be done prior when used
            with MultiMcmc.
        """
        self.read_to_write_memmap()
        mcmc_array = self.get_mcmc_array()
        mcmc.do_gibbs_sampling(
            mcmc_array, self.n_sample, self.rng, self.gibbs_weight,
            is_print=False)
        self.del_memmap()
        print("Location " + str(self.id) + " " + str(self.n_sample)
            + " samples")

    #override
    def resume_fitting(self, n_sample):
        """Overridden to use the common memmap between locations

        Also modify the print statements so it prints the progress when a
            location has completed their fitting rather than for each gibbs
            step.
        The extension of the memmap should be done prior when used with
            MultiMcmc.
        """
        if n_sample > self.n_sample:
            self.read_to_write_memmap()
            mcmc_array = self.get_mcmc_array()
            #in resume, do not use initial value as sample (False in arg 3)
            mcmc.do_gibbs_sampling(
                mcmc_array, n_sample - self.n_sample, self.rng,
                self.gibbs_weight, False, is_print=False)
            self.n_sample = n_sample
            self.del_memmap()
            print("Location " + str(self.id) + " " + str(self.n_sample)
                + " samples")

    #override
    def forecast(self, x, n_simulation, memmap_path, memmap_shape, i_space):
        """Overridden to use the common memmap between locations

        Args:
            x: model fields, eg array or pandas data frame
            n_simulation: number of Monte Carlo simulations
            memmap_path: location of the parent memmap file
            memmap_shape: shape of the parent memmap file
            i_space: pointer to the 0th dimension of the memmap to use, this
                should correspond to the location number, ie a number between 0
                and area_unmask.
        """
        self.read_memmap()
        if self.forecaster is None:
            self.forecaster = forecast.downscale.TimeSeriesForecaster(
                self, memmap_path, i_space)
            self.forecaster.start_forecast(n_simulation, x, memmap_shape)
        else:
            self.forecaster.memmap_path = memmap_path
            self.forecaster.resume_forecast(n_simulation, memmap_shape)
        self.del_memmap()

class MultiMcmc(object):
    """For handling Mcmc objects for each Gibbs component. Each Mcmc object is
        responsible for all locations so that each location shares the same
        memmap.

    Attributes:
        multi_series: MultiSeries object containing the array of TimeSeries
            objects
        mcmc_array: Mcmc object for each component
        mcmc_name_array: name for each Mcmc object in mcmc_array
        n_dim_total_array: numpy array, number of dimensions for each Mcmc in
        mcmc_array for all locations combined, ie n_dim_array * area_unmask
        n_dim_array: numpy array, number of dimensions for each Mcmc in
            mcmc_array for one location
        n_sample: number of MCMC samples
        memmap_dir: where to save the memmaps
    """

    def __init__(self, multi_series):
        self.multi_series = multi_series
        self.mcmc_array = []
        self.mcmc_name_array = []
        self.n_dim_total_array = None
        self.n_dim_array = None
        self.n_sample = multi_series.n_sample
        self.memmap_dir = multi_series.memmap_dir

        for time_series_i in self.multi_series.generate_unmask_time_series():
            time_series_i.n_sample = self.n_sample

        #look at the first time series, extract MCMC information
        self.n_dim_array = []
        self.n_dim_total_array = []
        iter = multi_series.generate_unmask_time_series()
        time_series_0 = iter.__next__()
        for mcmc_i in time_series_0.get_mcmc_array():
            self.n_dim_array.append(mcmc_i.n_dim)
            self.mcmc_array.append(mcmc.Mcmc(mcmc_i.dtype, mcmc_i.target))
            self.mcmc_name_array.append(mcmc_i.get_target_class())

        #assume each location has the same MCMC, get n_dim
        self.n_dim_array = np.asarray(self.n_dim_array, dtype=np.int32)
        self.n_dim_total_array = (
            self.n_dim_array * self.multi_series.area_unmask)

        #instantiate the memmap for each component
        for i, mcmc_i in enumerate(self.mcmc_array):
            mcmc_i.instantiate_memmap(
                self.memmap_dir, self.n_sample, self.n_dim_total_array[i])

    def extend_memmap(self, n_sample):
        """Extend the memmap for each Mcmc
        """
        if n_sample > self.n_sample:
            self.n_sample = n_sample
            for mcmc_i in self.mcmc_array:
                mcmc_i.extend_memmap(n_sample)

    def delete_old_memmap(self):
        """DANGEROUS: Actually deletes the file containing the old MCMC samples
        """
        for mcmc_i in self.mcmc_array:
            mcmc_i.delete_old_memmap()

    def read_to_write_memmap(self):
        """Read to write the memmap for each Mcmc
        """
        for mcmc_i in self.mcmc_array:
            mcmc_i.load_memmap("r+")

    def read_memmap(self):
        """Read the memmap for each Mcmc
        """
        for mcmc_i in self.mcmc_array:
            mcmc_i.load_memmap("r")

    def del_memmap(self):
        """Close the memmap for each Mcmc
        """
        for mcmc_i in self.mcmc_array:
            mcmc_i.del_memmap()

    def scatter_mcmc(self):
        """Make each Mcmc in the array of TimeSeries to have access the memmap
            in self.mcmc_array
        """
        area_unmask = self.multi_series.area_unmask
        for i, time_series_i in enumerate(
            self.multi_series.generate_unmask_time_series()):
            mcmc_array = time_series_i.get_mcmc_array()
            for j, mcmc_j in enumerate(mcmc_array):
                n_dim = self.n_dim_total_array[j]
                n_dim_i = self.n_dim_array[j]
                slice_index = slice(i*n_dim_i, (i+1)*n_dim_i)
                mcmc_j.set_memmap_slice(self.n_sample,
                                        n_dim,
                                        self.mcmc_array[j].memmap_path,
                                        slice_index)

class FitMessage(object):
    """Message for fitting for each location in parallel
    """
    def __init__(self, time_series):
        self.time_series = time_series
    def fit(self):
        self.time_series.fit()
        return self.time_series

class ResumeFittingMessage(object):
    """Message for resuming fitting for each location in parallel
    """
    def __init__(self, time_series, n_sample):
        self.time_series = time_series
        self.n_sample = n_sample
    def fit(self):
        self.time_series.resume_fitting(self.n_sample)
        return self.time_series

class ForecastMessage(object):
    """Message for forecasting each location in parallel
    """
    def __init__(self, time_series, model_field, n_simulation):
        self.time_series = time_series
        self.model_field = model_field
        self.n_simulation = n_simulation
    def forecast(self):
        return self.time_series.forecast(self.model_field, self.n_simulation)

class PlotMcmcMessage(object):
    """Message for printing the mcmc figures for each location in parallel
    """
    def __init__(self, downscale, chain, time_series, i_space, directory):
        self.time_series = time_series
        self.sub_directory = path.join(directory, str(time_series.id))
        if not path.isdir(self.sub_directory):
            os.mkdir(self.sub_directory)
    def print(self):
        self.time_series.print_mcmc(self.sub_directory)
