import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
from numpy import random

from compound_poisson import forecast
from compound_poisson import mcmc
from compound_poisson import time_series_mcmc

class MultiSeries(object):

    def __init__(self, data, n_arma=(0,0)):
        self.n_arma = n_arma
        self.time_series_array = []
        self.time_array = data.time_array
        self.model_field_units = data.model_field_units
        self.n_model_field = len(data.model_field)
        self.mask = data.mask
        self.parameter_mask_vector = []
        self.n_parameter = None
        self.n_total_parameter = None
        self.topography = data.topography
        self.shape = self.mask.shape
        self.area = self.shape[0] * self.shape[1]
        self.area_unmask = np.sum(np.logical_not(self.mask))
        self.n_sample = 10000
        self.burn_in = 0
        self.pool = None
        self.memmap_dir = ""
        self.forecaster = None
        self.mcmc = None

        #instantiate time series for every point in space
        #unmasked points have rain, provide it to the constructor to
            #TimeSeries
        #masked points do not have rain, cannot provide it
        time_series_array = self.time_series_array
        TimeSeries = self.get_time_series_class()
        for lat_i in range(self.shape[0]):
            time_series_array.append([])
            for long_i in range(self.shape[1]):

                x_i, rain_i = data.get_data(lat_i, long_i)
                is_mask = self.mask[lat_i, long_i]
                if is_mask:
                    #provide no rain if this space is masked
                    time_series = TimeSeries(x_i,
                                             poisson_rate_n_arma=n_arma,
                                             gamma_mean_n_arma=n_arma)
                else:
                    #provide rain
                    time_series = TimeSeries(x_i, rain_i.data, n_arma, n_arma)

                for i in range(time_series.n_parameter):
                    self.parameter_mask_vector.append(is_mask)
                self.n_parameter = time_series.n_parameter

                #provide information to time_series
                time_series.id = [lat_i, long_i]
                time_series_array[lat_i].append(time_series)

        #set other member variables
        self.set_seed_seq(random.SeedSequence())
        self.set_time_series_rng()
        self.parameter_mask_vector = np.asarray(self.parameter_mask_vector)
        self.n_total_parameter = self.area_unmask * self.n_parameter

    def get_time_series_class(self):
        return TimeSeriesMultiSeries

    def fit(self, pool=None):
        """Fit using Gibbs sampling

        Args:
            pool: optional, a pool object to do parallel tasks
        """
        if pool is None:
            pool = multiprocess.Serial()
        self.pool = pool
        self.instantiate_mcmc()
        self.scatter_mcmc_sample()

        message_array = []
        for time_series_i in self.generate_unmask_time_series():
            message_array.append(FitMessage(time_series_i))
        time_series_array = self.pool.map(FitMessage.fit, message_array)
        self.replace_unmask_time_series(time_series_array)

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
            self.n_sample = n_sample
            self.mcmc.extend_memmap(n_sample)
            self.scatter_mcmc_sample()

            message_array = []
            for time_series_i in self.generate_unmask_time_series():
                message_array.append(ResumeFittingMessage(time_series_i, n_sample))
            time_series_array = self.pool.map(FitMessage.fit, message_array)
            self.replace_unmask_time_series(time_series_array)


            self.delete_old_memmap()
        self.del_memmap()
        self.pool = None

    def instantiate_mcmc(self):
        """Instantiate MCMC objects
        """
        for time_series_i in self.generate_unmask_time_series():
            time_series_i.initalise_z()
            time_series_i.instantiate_mcmc()
        self.mcmc = MultiMcmc(self)

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

    def scatter_mcmc_sample(self):
        self.mcmc.scatter_mcmc()

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
        self.mcmc.read_memmap()

    def del_memmap(self):
        self.mcmc.del_memmap()

    def delete_old_memmap(self):
        self.mcmc.delete_old_memmap()

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

class TimeSeriesMultiSeries(time_series_mcmc.TimeSeriesHyperSlice):
    """Modify TimeSeriesSlice to only sample z
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
        #set these to None to ensure a memmap is not allocated, allocation is
            #done using MultiMcmc
        self.n_sample = None
        self.memmap_dir = None

    def fit(self):
        self.read_to_write_memmap()
        mcmc_array = self.get_mcmc_array()
        mcmc.do_gibbs_sampling(
            mcmc_array, self.n_sample, self.rng, self.gibbs_weight)
        self.del_memmap()

    def resume_fitting(self, n_sample):
        if n_sample > self.n_sample:
            self.read_to_write_memmap()
            mcmc_array = self.get_mcmc_array()
            #in resume, do not use initial value as sample (False in arg 3)
            mcmc.do_gibbs_sampling(
                mcmc_array, n_sample - self.n_sample, self.rng,
                self.gibbs_weight, False)
            self.n_sample = n_sample
            self.del_memmap()

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

class MultiMcmc(object):
    """
    Attributes:
        multi_series:
        mcmc_array: Mcmc object for each component
        mcmc_name_array: name for each Mcmc object in mcmc_array
        n_dim_array: numpy array, full dimension for each Mcmc in mcmc_array
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
    def __init__(self, time_series):
        self.time_series = time_series
    def fit(self):
        self.time_series.fit()
        return self.time_series

class ResumeFittingMessage(object):
    def __init__(self, time_series, n_sample):
        self.time_series = time_series
        self.n_sample
    def fit(self):
        self.time_series.resume_fitting(n_sample)
        return self.time_series

class PlotMcmcMessage(object):

    def __init__(self, downscale, chain, time_series, i_space, directory):
        self.time_series = time_series
        self.sub_directory = path.join(directory, str(time_series.id))
        if not path.isdir(self.sub_directory):
            os.mkdir(self.sub_directory)

    def print(self):
        self.time_series.print_mcmc(self.sub_directory)
