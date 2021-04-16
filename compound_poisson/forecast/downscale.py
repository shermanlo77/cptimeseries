"""Implementation of Forecaster for Downscale

Contain the classes compound_poisson.forecast.downscale.Forecaster

Note to future developers: only forecasting of the test set (future) has been
    implemented. Implementation of forecasting the training set should handle
    rng like time_series, a rng(s) for forecasting training set, another rng(s)
    for test set

compound_poisson.forecast.forecast_abstract.Forecaster
    <- Forecaster
        <- ForecasterGp

compound_poisson.forecast.time_series.Forecaster
    <- compound_poisson.forecast.downscale.TimeSeriesForecaster
because
compound_poisson.downscale.TimeSeriesDownscale
    <>1- compound_poisson.forecast.downscale.TimeSeriesForecaster

TODO: improve signature use of simulate_forecasts()
"""

from os import path

import numpy as np
from sklearn import gaussian_process

from compound_poisson.forecast import distribution_compare
from compound_poisson.forecast import forecast_abstract
from compound_poisson.forecast import roc
from compound_poisson.forecast import time_series


class Forecaster(forecast_abstract.Forecaster):
    """Contain Monte Carlo forecasts for Downscale

    Statistical note, method for forward simulation:
        Each location has a Forecaster object which simulates independently,
            in parallel and asynchronously. This means that each location will
            forward simulate using DIFFERENT MCMC samples. This was chosen
            because it is easier to parallise and it is faster. Effects average
            out but this means individual forecasts should not be used as each
            location would have different MCMC parameters.
        Please refer to commit 206aa73 to implement the forward forecast
            synchronously. That is: draw a MCMC sample, each location forward
                simulate using that MCMC sample. Draw another MCMC sample,
                simulate, draw, ...etc.

    Notes:
        self.data contains the test set, this includes the model fields AND the
            precipitation. This means the test set precipitation does not need
            to be passed when assessing the performance of the forecast

    Attributes:
        downscale: Downscale object, to forecast from
        data: the test set (dataset.Data object)
        forecast_array: memmap of forecasts
            dim 0: for each (unmasked) location
            dim 1: for each simulation
            dim 2: for each time point
    """

    def __init__(self, downscale, memmap_dir):
        self.downscale = downscale
        self.data = None
        super().__init__(memmap_dir)

    # override
    def make_memmap_path(self):
        super().make_memmap_path(type(self.downscale).__name__)

    # override
    # additional parameter data
    def start_forecast(self, n_simulation, data):
        """Start forecast simulations, to be called initially

        Args:
            n_simulation: number of simulations
            data: the test set (dataset.Data object)
        """
        self.data = data
        self.n_time = len(data)
        self.time_array = data.time_array
        super().start_forecast(n_simulation)

    # implemented
    def copy_to_memmap(self, memmap_to_copy):
        for i in range(len(self.forecast_array)):
            memmap_to_copy_i = memmap_to_copy[i]
            self.forecast_array[i, 0:len(memmap_to_copy_i)] = memmap_to_copy_i

    # implemented
    def simulate_forecasts(self, index_range, is_print=True):
        forecast_message = []
        for i_space, time_series_i in enumerate(
                self.downscale.generate_unmask_time_series()):

            # extract model fields for each unmasked time_series
            lat_i = time_series_i.id[0]
            long_i = time_series_i.id[1]
            x_i = self.data.get_model_field(lat_i, long_i)

            message = ForecastMessage(time_series_i,
                                      x_i,
                                      self.n_simulation,
                                      self.memmap_path,
                                      self.forecast_array.shape,
                                      i_space,
                                      is_print)
            forecast_message.append(message)

        time_series_array = self.downscale.pool.map(
            ForecastMessage.forecast, forecast_message)
        self.downscale.replace_unmask_time_series(time_series_array)

    # implemented
    def get_prob_rain(self, rain, index=None):
        """Get the probability if it will rain at least of a certian amount

        Args:
            rain: scalar, amount of rain to evaluate the probability
            index: time index (optional), otherwise, take ROC from all time
                points

        Return:
            matrix, dim 0 for each location, dim 1 for each time step
        """
        if index is None:
            index = slice(self.n_time)
        # for forecast_array...
        #     dim 0 is location
        #     dim 1 is for each simulation
        #     dim 2 is for each time point
        p_rain = np.mean(self.forecast_array[:, :, index] > rain, 1)
        return p_rain

    # override
    # to provide the shape of the memmap
    def load_memmap(self, mode):
        super().load_memmap(
            mode, (self.downscale.area_unmask, self.n_simulation, self.n_time))

    def load_locations_memmap(self, mode):
        """Call load_memmap() for each forecaster in self.downscale
        """
        for time_series_i in self.downscale.generate_unmask_time_series():
            time_series_i.forecaster.load_memmap(mode)

    def del_locations_memmap(self):
        """Call del_memmap() for each forecaster in self.downscale
        """
        for time_series_i in self.downscale.generate_unmask_time_series():
            time_series_i.forecaster.del_memmap()

    def generate_time_series_forecaster(self):
        """Generate the forecaster for every unmasked time series. Also load
            the memmap. Caution: ensure to call del_memmap() for each of the
            forecaster after use
        """
        for time_series_i in self.downscale.generate_unmask_time_series():
            forecaster = time_series_i.forecaster
            forecaster.load_memmap("r")
            yield time_series_i.forecaster

    def generate_forecaster_no_memmap(self):
        """Generate the forecaster for every unmasked time series, do not load
            memmap, used for parallel computation by delaying the calling of
            load_memap() at a later stage
        """
        for time_series_i in self.downscale.generate_unmask_time_series():
            yield time_series_i.forecaster

    # implemented
    def get_roc_curve_array(
            self, rain_warning_array, time_index=None, pool=None):
        """Get array of ROC curves

        Evaluate the ROC curve for different amounts of precipitation

        Args:
            rain_warning_array: array of amount of precipitation to be detected
            time_index: optional, a pointer (eg slice or array of indices) for
                time points to take ROC curve of
            pool: optional, used for parallel computing

        Return:
            array of roc.Roc objects which can be None if a value of
                precipitation in rain_warning_array was never observed
        """
        if time_index is None:
            time_index = slice(len(self.data))
        mask = self.data.mask
        observed_rain = self.data.rain[time_index, np.logical_not(mask)]
        # swap axes so that...
        #     dim 0: for each location
        #     dim 1: for each time point
        observed_rain = np.swapaxes(observed_rain, 0, 1)
        # when flatten, this is comparable with the return value from
        # self.get_prob_rain()
        observed_rain = observed_rain.flatten()

        roc_array = []
        # get roc curve for every rain_warning, else None if that amount of
        # rain was never observed in the test set
        for rain_warning in rain_warning_array:
            if np.any(rain_warning < observed_rain):
                p_rain = self.get_prob_rain(rain_warning, time_index).flatten()
                roc_curve = roc.Roc(rain_warning, p_rain, observed_rain, pool)
                roc_array.append(roc_curve)
            else:
                roc_array.append(None)
        return roc_array

    # implemented
    def compare_dist_with_observed(self, n_linspace=100):
        """Return an object from distribution_compare, used to compare the
            distribution of the precipitation of the forecast and the observed

        Args:
            observed_rain: numpy array of observed precipitation
            n_linspace: number of points to evaluate between 0 mm and max
                observed rain

        Return: distribution_compare.Downscale object
        """
        comparer = distribution_compare.Downscale()
        comparer.compare(self, n_linspace)
        return comparer


class ForecasterGp(Forecaster):
    """
    Do GP smoothing on the parameters for every forecast

    Take samples from the MCMC to construct a GP. For each forecast, the GP is
        sampled from and be used as parameters for the compound-Poisson model.

    Attributes:
        gp_input: array of spatial points (latitude, longitude) of the
            trained model
        gp_array: array of GaussianProcessRegressor() objects, one for each
            parameter
    """

    def __init__(self, downscale, memmap_dir, topo_key):
        """Constructor

        Args:
            downscale: parent Downscale object
            memmap_dir: location of the forecast memmap
            topo_key: array of topography keys to use as gp inputs, eg
                ["latitude", "longitude"]
        """
        super().__init__(downscale, memmap_dir)
        self.gp_input = None
        self.gp_array = []

        area_unmask = downscale.area_unmask
        # get the input variables of the GP from the topography information
        topo_dic = downscale.topography

        n_sample = 100  # number of posterior samples to use
        # each location has multiple samples of beta, fit gp onto a sample of
        # samples
        # gp_input:
        #     dim 0: for each location
        #     dim 1: for each topo key
        self.gp_input = np.zeros((area_unmask, len(topo_key)))
        # gp_output: array of numpy for each parameter (eg reg for temperature)
        # each element has dimensions, dim 0: for each location, dim 1: for
        # each gp sample
        gp_output = []
        for i_parameter in range(downscale.n_parameter):
            gp_output.append(np.zeros((area_unmask, n_sample)))

        # get topography information for each location
        for i_location, time_series_i in enumerate(
                downscale.generate_unmask_time_series()):
            coordinates = time_series_i.id
            for i_key, key in enumerate(topo_key):
                self.gp_input[i_location, i_key] = (
                    topo_dic[key][coordinates[0], coordinates[1]])
            # ensure time_series does not set from the posterior sample, these
            # are set manually from the gp
            time_series_i.set_from_mcmc = False

        # select random mcmc samples
        rng = downscale.rng
        mcmc_index_array = rng.choice(
            range(downscale.burn_in, downscale.n_sample), n_sample)

        # extract parameters to fit onto
        for i_mcmc, mcmc_index in enumerate(mcmc_index_array):
            # set mcmc sample
            for time_series_i in downscale.generate_unmask_time_series():
                time_series_i.read_memmap()
                time_series_i.set_parameter_from_sample_i(mcmc_index)
                time_series_i.del_memmap()
            # extract parameter and save it to gp_output
            parameter_vector = downscale.get_parameter_vector().copy()
            for i_parameter in range(downscale.n_parameter):
                slice_index = slice(
                    i_parameter*area_unmask, (i_parameter+1)*area_unmask)
                parameter_i = parameter_vector[slice_index]
                gp_output[i_parameter][:, i_mcmc] = parameter_i

        # fit gp for each parameter
        for i_parameter in range(downscale.n_parameter):
            gp = GaussianProcessRegressor()
            gp.fit(self.gp_input, gp_output[i_parameter])
            # do not save x_train, this is set in forecasting, prevents saving
            # duplicates of this variable onto disk
            gp.delete_x_train()
            self.gp_array.append(gp)

    # override
    def simulate_forecasts(self, index_range):
        # loop: get mcmc sample, forecast, ..etc
        # the forecast is done in parallel
        area_unmask = self.downscale.area_unmask
        n_parameter = self.downscale.n_parameter

        # set x_train for each gp, they all point to self.gp_input, prevents
        # deep copies
        for gp_i in self.gp_array:
            gp_i.set_x_train(self.gp_input)

        for i_forecast in index_range:

            # get the parameter and smooth it using GP
            parameter_vector = np.zeros(area_unmask * n_parameter)
            for i_parameter in range(n_parameter):
                slice_index = slice(
                    i_parameter*area_unmask, (i_parameter+1)*area_unmask)
                parameter_i = self.gp_array[i_parameter].sample_y_at_train(
                    self.downscale.rng)
                parameter_vector[slice_index] = parameter_i.flatten()

            # set the smoothed parameter
            self.downscale.set_parameter_vector(parameter_vector)
            self.downscale.update_all_cp_parameters()
            # do one sample
            #
            # hack: set the member variable n_simulation to do only one
            # forecast (could be improved here)
            self.n_simulation = i_forecast + 1
            super().simulate_forecasts([i_forecast], False)

            print("Predictive sample", i_forecast)

        # remove x_train so that duplicates are not saved
        for gp_i in self.gp_array:
            gp_i.delete_x_train()


class GaussianProcessRegressor(gaussian_process.GaussianProcessRegressor):
    """Modification of sklearn.gaussian_process.GaussianProcessRegressor

    Custom modification of sklearn.gaussian_process.GaussianProcessRegressor to
        accept multiple independent samples of the output for a given input.
        In other words, multiple observations of the response for a given
        explanatory variable. It is designed to reduce the size of the kernel
        matrix compared to stacking multiple observations.
    How to use: pass a design matrix X (n_sample x n_features) and a response
        design matrix (n_sample x n_observations) to the method fit(). Methods
        as such predict() and sample_y() now return a response vector, merging
        the n_targets (using notation in the orginial source in the method
        fit()) together.

    Modifications:
        - The method fit() uses the parameters differently
        - The method fit() modifies the member variables further
        - The member variable copy_X_train = False by default
        - Aims to elimate the member variable y_train_ because it appears it is
            not used anywhere in prediction
        - Added new methods

    Unmodified copyright notices from the originial source:
        Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
        Modified by: Pete Green <p.l.green@liverpool.ac.uk>
        License: BSD 3 clause
    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
        for further information on the orginial source
    """

    def __init__(self):
        super().__init__()
        self.copy_X_train = False

    # override
    def fit(self, X, y):
        super().fit(X, y)
        self.alpha_ = np.mean(self.alpha_, axis=1)
        self.y_train_ = None

    def delete_x_train(self):
        """Set the member variable X_train_ to be none, this prevents saving
             duplicates of this member variable (there are multiple gps using
             the same trainig set) to disk. Designed as a precaution.
        """
        self.X_train_ = None

    def set_x_train(self, x_train):
        """Set the member variable X_train_ to something after calling
            delete_x_train(). X_train_ has dimensions, dim 0: each location,
            dim 1: each feature
        """
        self.X_train_ = x_train

    def sample_y_at_train(self, rng):
        """Sample the GP at the training set
        """
        if self.X_train_ is None:
            raise Exception("Must set X_train_")
        return super().sample_y(self.X_train_, random_state=rng)


class TimeSeriesForecaster(time_series.Forecaster):
    """Used by TimeSeriesDownscale class

    Extended to handle MCMC samples in memmaps. The member variable
        forecast_array is shared with all spatial points and is handled by the
        corresponding Downscale object.

    Attributes:
        i_space: pointer for space, or the 0th dimension for the forecast_array
        mememap_path: location of the forecast_array
        memmap_shape: shape of the forecast_array
    """

    def __init__(self, time_series, memmap_path, i_space):
        super().__init__(time_series, path.dirname(memmap_path))
        self.i_space = i_space
        self.memmap_path = memmap_path
        self.memmap_shape = None

    # override
    def start_forecast(self, n_simulation, model_field, memmap_shape):
        """Start forecast simulations, to be called initially

        Override as a memmap does not need to be created to store the
            forecasts, this has already been done by the corresponding
            Downscale object.

        Args:
            n_simulation: number of simulations
            model_field: model fields for test set
            memmap_shape: shape of the forecast_array
        """
        self.memmap_shape = memmap_shape
        super().start_forecast(n_simulation, model_field)

    # override
    def make_memmap_path(self):
        """Do nothing, memmap_path has already been provided
        """
        pass

    # override
    def simulate_forecasts(self, index_range):
        # do not print progress
        super().simulate_forecasts(index_range, False)

    # override
    def resume_forecast(self, n_simulation, memmap_shape):
        """Simulate more forecasts

        Override as the handling of forecast_array is done already by the
            corresponding Downscale object.

        Args:
            n_simulation: total amount of simulations, ie should be higher than
                previous
            memmap_shape: shape of the forecast_array
        """
        if n_simulation > self.n_simulation:
            self.memmap_shape = memmap_shape
            n_simulation_old = self.n_simulation
            self.n_simulation = n_simulation
            self.load_memmap("r+")
            # False in argument to not print progress
            self.simulate_forecasts(range(n_simulation_old, self.n_simulation))
            self.del_memmap()

    # override
    def load_memmap(self, mode):
        """Load the memmap file for forecast_array

        Override to use the forcast_array provided by Downscale. This is shared
            with all spatial points so extract the corresponding slice.

        Args:
            mode: not used, force "r+", prevent a "w+" because the memmap is
                already created
        """
        mode = "r+"
        super().load_memmap(mode, self.memmap_shape)
        self.forecast_array = self.forecast_array[self.i_space]


class ForecastMessage(object):
    """Message to forecast all spatial points in parallel
    """

    def __init__(self,
                 time_series,
                 model_field,
                 n_simulation,
                 memmap_path,
                 memmap_shape,
                 i_space,
                 is_print):
        self.time_series = time_series
        self.model_field = model_field
        self.n_simulation = n_simulation
        self.memmap_path = memmap_path
        self.memmap_shape = memmap_shape
        self.i_space = i_space
        self.is_print = is_print

    def forecast(self):
        self.time_series.forecast(
            self.model_field, self.n_simulation,
            self.memmap_path, self.memmap_shape, self.i_space)
        if self.is_print:
            print("Predictive location", self.i_space)
        return self.time_series
