import math
import pathlib

import numpy as np
from numpy import random

import compound_poisson
from compound_poisson import mcmc
from compound_poisson import multiprocess
from compound_poisson import time_series_mcmc
from compound_poisson.mcmc import target_downscale
from compound_poisson.mcmc import target_model_field

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
        model_field_shift: mean of model field, vector, entry for each model
            field
        model_field_scale: std of model field, vector, entry of reach model
            field
        square_error: matrix (area_unmask x area_unmask) containing square error
            of topography between each point in space
        pool: object for parallel programming
        memmap_path: location to store mcmc samples
    """

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
        self.topography_normalise = data.topography_normalise
        self.shape = self.mask.shape
        self.area = self.shape[0] * self.shape[1]
        self.area_unmask = np.sum(np.logical_not(self.mask))
        self.seed_seq = random.SeedSequence()
        self.rng = None
        self.parameter_target = None
        self.parameter_gp_target = None
        self.parameter_mcmc = None
        self.parameter_gp_mcmc = None
        self.z_mcmc = None
        self.n_sample = 10000
        self.model_field_shift = []
        self.model_field_scale = []
        self.square_error = np.zeros((self.area_unmask, self.area_unmask))
        self.pool = None
        self.memmap_path = pathlib.Path(__file__).parent.absolute()

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
                    #provide no rain
                    time_series = TimeSeriesDownscale(
                        x_i, poisson_rate_n_arma=n_arma,
                        gamma_mean_n_arma=n_arma)
                else:
                    #provide rain
                    time_series = TimeSeriesDownscale(
                        x_i, rain_i.data, n_arma, n_arma)
                #provide information to time_series
                time_series.id = [lat_i, long_i]
                time_series.x_shift = self.model_field_shift
                time_series.x_scale = self.model_field_scale
                time_series.time_array = self.time_array
                time_series.memmap_path = self.memmap_path
                time_series_array[lat_i].append(time_series)
                for i in range(time_series.n_parameter):
                    self.parameter_mask_vector.append(is_mask)
                self.n_parameter = time_series.n_parameter

        #set other member variables
        self.set_rng(self.seed_seq)
        self.parameter_mask_vector = np.asarray(self.parameter_mask_vector)
        self.n_total_parameter = self.area_unmask * self.n_parameter
        self.parameter_target = target_downscale.TargetParameter(self)
        self.parameter_gp_target = target_downscale.TargetGp(self)

    def fit(self):
        """Fit using Gibbs sampling
        """
        self.pool = multiprocess.Pool()
        self.initalise_z()
        self.instantiate_mcmc()
        mcmc_array = self.get_mcmc_array()
        mcmc.do_gibbs_sampling(mcmc_array, self.n_sample, self.rng)
        self.pool.close()
        self.pool.join()
        self.pool = None

    def instantiate_mcmc(self):
        """Instantiate MCMC objects
        """
        self.parameter_mcmc = mcmc.Elliptical(
            self.n_sample, self.memmap_path, self.parameter_target, self.rng)
        self.parameter_gp_mcmc = mcmc.Rwmh(
            self.n_sample, self.memmap_path, self.parameter_gp_target, self.rng)
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
        #reload memmap as parallel deep copy the memmap
        self.read_to_write_z_memmap()
        self.update_all_cp_parameters()

    def set_rng(self, seed_sequence):
        """Set rng for all time series

        Args:
            seed_sequence: numpy.random.SeedSequence object
        """
        self.seed_seq = seed_sequence
        self.rng = self.spawn_rng()
        for time_series in self.generate_all_time_series():
            time_series.rng = self.spawn_rng()

    def set_memmap_path(self, memmap_path):
        """Set the location to save mcmc sample onto disk

        Modifies all mcmc objects to save the mcmc sample onto a specified
            location
        """
        self.memmap_path = memmap_path
        for time_series in self.generate_all_time_series():
            time_series.memmap_path = memmap_path

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
        self.read_to_write_z_memmap()

    def get_log_likelihood(self):
        """Return log likelihood
        """
        method = (compound_poisson.time_series.TimeSeries
            .get_joint_log_likelihood)
        ln_l_array = self.pool.map(method, self.generate_unmask_time_series())
        return np.sum(ln_l_array)

    def forecast(self, data, n_simulation):
        """Do forecast

        Args:
            data: test data
            n_simulation: number of Monte Carlo simulations

        Return:
            nested array of Forecast objects, shape corresponding to fine grid
        """
        #contains forecast objects for each unmasked time series
        forecast_array = []
        area_unmask = self.area_unmask
        n_total_parameter = self.n_total_parameter
        #do forecasting for all unmaksed time series
        for i, time_series in enumerate(self.generate_unmask_time_series()):
            lat_i = time_series.id[0]
            long_i = time_series.id[1]
            x_i = data.get_model_field(lat_i, long_i)
            #extract mcmc chain corresponding to this location
            parameter_mcmc = np.array(self.parameter_mcmc.sample_array)
            parameter_mcmc = parameter_mcmc[
                :, range(i, n_total_parameter, area_unmask)]
            time_series.parameter_mcmc = parameter_mcmc
            forecast = time_series.forecast(x_i, n_simulation)
            forecast_array.append(forecast)

        #convert array into nested array (2D array), use None for masked time
            #series
        forecast_nested_array = []
        i = 0
        for lat_i in range(self.shape[0]):
            forecast_nested_array_i = []
            for long_i in range(self.shape[1]):
                is_mask = self.mask[lat_i, long_i]
                if self.mask[lat_i, long_i]:
                    forecast = None
                else:
                    forecast = forecast_array[i]
                    i += 1
                forecast_nested_array_i.append(forecast)
            forecast_nested_array.append(forecast_nested_array_i)

        return forecast_nested_array

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

    def read_memmap(self):
        """Set memmap objects to read from file

        Required when loading saved object from joblib
        """
        for mcmc in self.get_mcmc_array():
            mcmc.read_memmap()

    def read_to_write_z_memmap(self):
        """Set memmap objects for z mcmc to read and write from file

        Required because parallel does a deep copy of time_series objects which
            contain ZMcmc objects. When gather is called, memmap objects will
            need to be instantiated again
        """
        if not self.z_mcmc is None:
            self.z_mcmc.read_to_write_memmap()

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

class DownscaleDual(Downscale):
    """Collection of multiple TimeSeries objects

    Fit a compound Poisson time series on multiple locations in 2d space.
        Parameters have a Gaussian process (GP) prior with gamma hyper
        parameters. Model fields on the coarse grid have a GP prior, model
        fields on the fine grid can be sampled

    See superclass Downscale

    Attributes:
        model_field_coarse: dictionary model fields on the coarse grid for each
            time point
        topography_coarse_normalise: dictionary of normalised topography
            information on the coarse grid
        n_coarse: number of points on the coarse grid
        model_field_target: compound_poisson.mcmc.target_model_field
            .TargetModelFieldArray object
        model_field_mcmc: mcmc object wrapping model_field_target
        model_field_gp_target: compound_poisson.mcmc.target_model_field
            .TargetGp object
        model_field_gp_mcmc: mcmc object wrapping model_field_gp_target
    """

    def __init__(self, data, n_arma=(0,0)):
        super().__init__(data, n_arma)
        self.model_field_coarse = data.model_field_coarse
        self.topography_coarse_normalise = data.topography_coarse_normalise
        self.n_coarse = None
        self.model_field_target = None
        self.model_field_mcmc = None
        self.model_field_gp_target = None
        self.model_field_gp_mcmc = None

        for model_field in self.model_field_coarse.values():
            self.n_coarse = model_field[0].size
            break

        self.model_field_target = target_model_field.TargetModelFieldArray(self)
        self.model_field_gp_target = target_model_field.TargetGp(self)

    def instantiate_mcmc(self):
        """Instantiate MCMC objects
        """
        super().instantiate_mcmc()
        self.model_field_mcmc = mcmc.Elliptical(
            self.n_sample, self.memmap_path, self.model_field_target, self.rng)
        self.model_field_gp_mcmc = mcmc.Rwmh(self.n_sample,
                                             self.memmap_path,
                                             self.model_field_gp_target,
                                             self.rng)
        #ordering is important, calculate the kernel matrices then posterior
            #gaussian process mean
        self.model_field_gp_target.save_cov_chol()
        self.model_field_target.update_mean()

    def get_mcmc_array(self):
        mcmc_array = super().get_mcmc_array()
        mcmc_array.append(self.model_field_mcmc)
        mcmc_array.append(self.model_field_gp_mcmc)
        return mcmc_array

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

    def instantiate_mcmc(self):
        """Instantiate all MCMC objects

        Override
        Only instantiate slice sampling for z
        """
        self.parameter_mcmc = None
        self.z_mcmc = mcmc.ZSlice(
            self.n_sample, self.memmap_path, self.z_target, self.rng)

    def print_chain_property(self, directory):
        """Override - only print chain for z
        """
        plt.figure()
        plt.plot(np.asarray(self.z_mcmc.slice_width_array))
        plt.ylabel("Latent variable slice width")
        plt.xlabel("Latent variable sample number")
        plt.savefig(path.join(directory, "slice_width_z.pdf"))
        plt.close()
