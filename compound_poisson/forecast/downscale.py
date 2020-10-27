"""Implementation of Forecaster for Downscale

Contain the classes compound_poisson.forecast.downscale.Forecaster

Note to future developers: only forecasting of the test set (future) has been
    implemented. Implementation of forecasting the training set should handle
    rng like time_series, a rng(s) for forecasting training set, another rng(s)
    for test set

compound_poisson.forecast.forecast_abstract.Forecaster
    <- compound_poisson.forecast.downscale.Forecaster

compound_poisson.forecast.time_series.Forecaster
    <- compound_poisson.forecast.downscale.TimeSeriesForecaster
because
compound_poisson.downscale.TimeSeriesDownscale
    <>1- compound_poisson.forecast.downscale.TimeSeriesForecaster
"""

from os import path

import numpy as np

from compound_poisson import roc
from compound_poisson.forecast import distribution_compare
from compound_poisson.forecast import forecast_abstract
from compound_poisson.forecast import time_series

class Forecaster(forecast_abstract.Forecaster):
    """Contain Monte Carlo forecasts for Downscale

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

    #override
    def make_memmap_path(self):
        super().make_memmap_path(type(self.downscale).__name__)

    #override
        #additional parameter data
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

    #implemented
    def copy_to_memmap(self, memmap_to_copy):
        for i in range(len(self.forecast_array)):
            memmap_to_copy_i = memmap_to_copy[i]
            self.forecast_array[i, 0:len(memmap_to_copy_i)] = memmap_to_copy_i

    #implemented
    def simulate_forecasts(self, index_range, is_print=True):
        area_unmask = self.downscale.area_unmask
        n_total_parameter = self.downscale.n_total_parameter

        forecast_message = []
        for i_space, time_series in enumerate(
            self.downscale.generate_unmask_time_series()):

            #extract model fields for each unmasked time_series
            lat_i = time_series.id[0]
            long_i = time_series.id[1]
            x_i = self.data.get_model_field(lat_i, long_i)

            message = ForecastMessage(time_series,
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

    #implemented
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
        #for forecast_array...
            #dim 0 is location
            #dim 1 is for each simulation
            #dim 2 is for each time point
        p_rain = np.mean(self.forecast_array[:, :, index] > rain, 1)
        return p_rain

    #override
        #to provide the shape of the memmap
    def load_memmap(self, mode):
        super().load_memmap(
            mode, (self.downscale.area_unmask, self.n_simulation, self.n_time))

    def load_locations_memmap(self, mode):
        """Call load_memmap() for each forecaster in self.downscale
        """
        for time_series in self.downscale.generate_unmask_time_series():
            time_series.forecaster.load_memmap(mode)

    def del_locations_memmap(self):
        """Call del_memmap() for each forecaster in self.downscale
        """
        for time_series in self.downscale.generate_unmask_time_series():
            time_series.forecaster.del_memmap()

    def generate_time_series_forecaster(self):
        """Generate the forecaster for every unmasked time series. Also load the
            memmap. Caution: ensure to call del_memmap() for each of the
            forecaster after use
        """
        for time_series in self.downscale.generate_unmask_time_series():
            forecaster = time_series.forecaster
            forecaster.load_memmap("r")
            yield time_series.forecaster

    def generate_forecaster_no_memmap(self):
        """Generate the forecaster for every unmasked time series, do not load
            memmap, used for parallel computation by delaying the calling of
            load_memap() at a later stage
        """
        for time_series in self.downscale.generate_unmask_time_series():
            forecaster = time_series.forecaster
            yield time_series.forecaster

    #implemented
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
        #swap axes so that...
            #dim 0: for each location
            #dim 1: for each time point
        observed_rain = np.swapaxes(observed_rain, 0, 1)
        #when flatten, this is comparable with the return value from
            #self.get_prob_rain()
        observed_rain = observed_rain.flatten()

        roc_array = []
        #get roc curve for every rain_warning, else None if that amount of rain
            #was never observed in the test set
        for rain_warning in rain_warning_array:
            if np.any(rain_warning < observed_rain):
                p_rain = self.get_prob_rain(rain_warning, time_index).flatten()
                roc_curve = roc.Roc(rain_warning, p_rain, observed_rain, pool)
                roc_array.append(roc_curve)
            else:
                roc_array.append(None)
        return roc_array

    #implemented
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

    #override
    def start_forecast(self, n_simulation, model_field, memmap_shape):
        """Start forecast simulations, to be called initially

        Override as a memmap does not need to be created to store the forecasts,
            this has already been done by the corresponding Downscale object.

        Args:
            n_simulation: number of simulations
            model_field: model fields for test set
            memmap_shape: shape of the forecast_array
        """
        self.memmap_shape = memmap_shape
        super().start_forecast(n_simulation, model_field)

    #override
    def make_memmap_path(self):
        """Do nothing, memmap_path has already been provided
        """
        pass

    #override
    def simulate_forecasts(self, index_range):
        #do not print progress
        super().simulate_forecasts(index_range, False)

    #override
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
            #False in argument to not print progress
            self.simulate_forecasts(range(n_simulation_old, self.n_simulation))
            self.del_memmap()

    #override
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
        self.time_series.forecast(self.model_field, self.n_simulation,
            self.memmap_path, self.memmap_shape, self.i_space)
        if self.is_print:
            print("Predictive location", self.i_space)
        return self.time_series
