"""Forecaster implementation for TimeSeries. Contain the classes
    compound_poisson.forecast.time_series.Forecaster
    compound_poisson.forecast.time_series.SelfForecaster

compound_poisson.forecast.forecast_abstract.Forecaster
    <- compound_poisson.forecast.time_series.Forecaster
        <- compound_poisson.forecast.time_series.SelfForecaster
"""

import numpy as np
from scipy import stats

from compound_poisson.forecast import distribution_compare
from compound_poisson.forecast import forecast_abstract
from compound_poisson.forecast import roc


class Forecaster(forecast_abstract.Forecaster):
    """Contain Monte Carlo forecasts

    Used by the methods TimeSeries.forecast() and TimeSeries.forecast_self().

    Attributes:
        time_series: pointer to parent TimeSeries object
        model_field: stored model fields of the test set
        forecast_array: memmap of forecasts
            dim 0: for each simulation
            dim 1: for each time point
        forecast: expectation of all forecasts, array
        forecast_median: median of all forecasts
        forecast_sigma: dictionary of sigma errors (quantiles) of all
            forecasts,
            keys are [-3, -2, -1, 0, 1, 2, 3] which correspond to the sigma
            level
        forecast_quartile: 3 array, each containing array of 25%, 50%, 75%
            quantile forecasts
    """

    def __init__(self, time_series, memmap_dir):
        self.time_series = time_series
        self.model_field = None
        self.forecast = None
        self.forecast_median = None
        self.forecast_sigma = {}
        self.forecast_quartile = [[], [], []]
        super().__init__(memmap_dir)

    # override
    def make_memmap_path(self):
        super().make_memmap_path(type(self.time_series).__name__)

    # override
    # additional parameter model_field to store the test set
    def start_forecast(self, n_simulation, model_field=None):
        self.model_field = model_field
        if model_field is None:
            self.n_time = len(self.time_series)
        else:
            self.n_time = len(model_field)
        super().start_forecast(n_simulation)

    # implemented
    def copy_to_memmap(self, memmap_to_copy):
        self.forecast_array[0:len(memmap_to_copy)] = memmap_to_copy[:]

    # implemented
    def simulate_forecasts(self, index_range, is_print=True):
        for i in index_range:
            forecast_i = self.get_simulated_forecast()
            self.forecast_array[i] = forecast_i.y_array
            if is_print:
                print("Predictive sample", i)
        self.time_array = forecast_i.time_array
        self.get_forecast()

    # implemented
    def get_prob_rain(self, rain):
        """Get the probability if it will rain at least of a certian amount

        Args:
            rain: scalar, amount of rain to evaluate the probability

        Return:
            vector, a probability for each day
        """
        p_rain = np.mean(self.forecast_array > rain, 0)
        return p_rain

    # override
    # shape for memmap provided
    def load_memmap(self, mode, memmap_shape=None):
        """Load the memmap file for forecast_array

        Args:
            mode: how to read the memmap file, eg "w+", "r+", "r"
            memmap_shape: the shape of forecast_array. Provide None is
                forecasting the training set
        """
        if memmap_shape is None:
            super().load_memmap(mode, (self.n_simulation, self.n_time))
        else:
            super().load_memmap(mode, memmap_shape)

    def get_simulated_forecast(self):
        """Return a TimeSeries object with simulated values
        """
        forecast_i = self.time_series.instantiate_forecast(self.model_field)
        forecast_i.simulate()
        return forecast_i

    def get_forecast(self):
        """Calculate statistics over all the provided forecasts
        """
        self.forecast = np.mean(self.forecast_array, 0)
        sigma_array = range(-3, 4)
        # work out quantiles for forecast_sigma and forecast_quartile together
        quantiles = np.concatenate((stats.norm.cdf(sigma_array), [0.25, 0.75]))
        forecast_quantile = np.quantile(self.forecast_array, quantiles, 0)
        for i in range(len(sigma_array)):
            self.forecast_sigma[sigma_array[i]] = forecast_quantile[i]
        self.forecast_median = self.forecast_sigma[0]
        self.forecast_quartile[0] = forecast_quantile[len(sigma_array)]
        self.forecast_quartile[1] = self.forecast_median
        self.forecast_quartile[2] = forecast_quantile[len(sigma_array)+1]

    # implemented
    def get_roc_curve(self, rain_warning, rain_true, time_index=None):
        """Return a ROC curve

        Args:
            rain_warning: the amount of precipitation to be detected
            rain_true: observed precipitation, array, for each time point
            time_index: slice object, which time points to consider

        Return:
            roc.Roc object, other None is returned if rain larger than
                rain_warning was never observed
        """
        if np.any(rain_true > rain_warning):
            p_rain_warning = self.get_prob_rain(rain_warning)
            if time_index is not None:
                p_rain_warning = p_rain_warning[time_index]
                rain_true = rain_true[time_index]
            roc_curve = roc.Roc(rain_warning, p_rain_warning, rain_true)
        else:
            roc_curve = None
        return roc_curve

    # implemented
    def get_roc_curve_array(
            self, rain_warning_array, rain_observed, time_index=None):
        """Get array of ROC curves

        Evaluate the ROC curve for different amounts of precipitation

        Args:
            rain_warning_array: array of amount of precipitation to be detected
            rain_observed: observed precipitation, array, for each time point
            time_index: optional, a pointer (eg slice or array of indices) for
                time points to take ROC curve of

        Return:
            array of roc.Roc objects which can be None if a value of
                precipitation in rain_warning_array was never observed
        """
        roc_array = []
        for rain_warning in rain_warning_array:
            roc_curve = self.get_roc_curve(
                rain_warning, rain_observed, time_index)
            roc_array.append(roc_curve)
        return roc_array

    # implemented
    def compare_dist_with_observed(
            self, observed_rain, n_linspace=500):
        """Return an object from distribution_compare, used to compare the
            distribution of the precipitation of the forecast and the observed

        Args:
            observed_rain: numpy array of observed precipitation
            n_linspace: number of points to evaluate between 0 mm and max
                observed rain

        Return: distribution_compare.TimeSeries object
        """
        comparer = distribution_compare.TimeSeries()
        comparer.compare(self, observed_rain, n_linspace)
        return comparer

    def bootstrap(self, rng):
        """Return a bootstrapped forecast array of itself

        Should be used for plotting or sensitivity analysis only

        sample_array may not guarantee to be a deep copy of memmap, investigate
            further if you require modifying the bootstrapped object.
        """
        bootstrap = Forecaster(self.time_series, self.memmap_dir)
        bootstrap.time_array = self.time_array
        bootstrap.forecast_array = self.forecast_array[
            rng.randint(self.n_simulation, size=self.n_simulation), :]
        bootstrap.get_forecast()
        return bootstrap

    def __getitem__(self, index):
        """
        Args:
            index: slice object
        """
        # only to be used for plotting purposes
        # does not copy model fields
        slice_copy = Forecaster(self.time_series, self.memmap_dir)
        slice_copy.time_array = self.time_array[index]
        if self.forecast_array is None:
            slice_copy.forecast_array = None
        else:
            slice_copy.forecast_array = self.forecast_array[:, index]
        slice_copy.forecast = self.forecast[index]
        slice_copy.forecast_median = self.forecast_median[index]
        slice_copy.forecast_sigma = {}
        for key, forecast_sigma_i in self.forecast_sigma.items():
            slice_copy.forecast_sigma[key] = forecast_sigma_i[index]
        slice_copy.forecast_quartile = []
        for quartile in self.forecast_quartile:
            slice_copy.forecast_quartile.append(quartile[index])
        slice_copy.n_time = len(slice_copy.time_array)
        slice_copy.n_simulation = self.n_simulation
        slice_copy.memmap_path = self.memmap_path
        return slice_copy


class SelfForecaster(Forecaster):
    """For forecasting the training set

    Different as the z were estimated in MCMC
    """

    def __init__(self, time_series, memmap_dir):
        super().__init__(time_series, memmap_dir)

    # override
    def start_forecast(self, n_simulation):
        # implemented in such a way it passes no model fields
        super().start_forecast(n_simulation)

    # override
    def get_simulated_forecast(self):
        """Return a TimeSeries object with simulated values, with z known
        """
        forecast_i = self.time_series.instantiate_forecast_self()
        forecast_i.simulate_given_z()
        return forecast_i
