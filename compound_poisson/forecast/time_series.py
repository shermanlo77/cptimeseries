import datetime
import math
import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from compound_poisson import roc

class Forecaster(object):
    """Contain Monte Carlo forecasts

    Contain Monte Carlo forecasts in forecast_array as an array a numpy.memmap
        object. Add forecasts using the method start_forecast(). To add more
        forecasts after that, call the method resume_forecast(). Summary
        statistics are stored in member variables,
    Used by the methods TimeSeries.forecast() and TimeSeries.forecast_self().

    Attributes:
        time_series: pointer to parent TimeSeries object
        time_array: array, containing time stamps for each point in the forecast
        model_field: stored model fields of the test set
        forecast_array: memmap of forecasts
            dim 0: for each simulation
            dim 1: for each time point
        forecast: expectation of all forecasts, array
        forecast_median: median of all forecasts
        forecast_sigma: dictionary of sigma errors (quantiles) of all forecasts,
            keys are [-3, -2, -1, 0, 1, 2, 3] which correspond to the sigma
            level
        n_time: length of forecast or number of time points
        n_simulation: number of simulations to be done
        memmap_dir: directory to forecast_array memmap file
        memmap_path: file path to forecast_array memmap file
    """

    def __init__(self, time_series, memmap_dir):
        self.time_series = time_series
        self.time_array = None
        self.model_field = None
        self.forecast_array = None
        self.forecast = None
        self.forecast_median = None
        self.forecast_sigma = {}
        self.n_time = None
        self.n_simulation = 0
        self.memmap_dir = memmap_dir
        self.memmap_path = None

    def make_memmap_path(self):
        """Make a new memmap file name

        Uses datetime to make it unique and identifiable
        """
        datetime_id = str(datetime.datetime.now())
        datetime_id = datetime_id.replace("-", "")
        datetime_id = datetime_id.replace(":", "")
        datetime_id = datetime_id.replace(" ", "")
        datetime_id = datetime_id[0:14]
        file_name = ("_" + type(self).__name__ + type(self.time_series).__name__
            + datetime_id + ".dat")
        self.memmap_path = path.join(self.memmap_dir, file_name)

    def start_forecast(self, n_simulation, model_field=None):
        """Start forecast simulations, to be called initially

        Args:
            n_simulation: number of simulations
            model_field: model fields for test set (None is test set is training
                set)
        """
        self.model_field = model_field
        if model_field is None:
            self.n_time = len(self.time_series)
        else:
            self.n_time = len(model_field)
        self.n_simulation = n_simulation
        self.make_memmap_path()
        self.load_memmap("w+")
        self.simulate_forecasts(range(n_simulation))

    def simulate_forecasts(self, index_range):
        """Simulate forecasts

        Simulate the forecasts and save results to self.forecast_array

        Args:
            index_range: array of points to save results onto forecast_array
        """
        for i in index_range:
            print("Predictive sample", i)
            forecast_i = self.get_simulated_forecast()
            self.forecast_array[i] = forecast_i.y_array
        self.time_array = forecast_i.time_array
        self.get_forecast()
        self.del_memmap()

    def get_simulated_forecast(self):
        """Return a TimeSeries object with simulated values
        """
        forecast_i = self.time_series.instantiate_forecast(self.model_field)
        forecast_i.simulate()
        return forecast_i

    def resume_forecast(self, n_simulation):
        """Simulate more forecasts

        Args:
            n_simulation: total amount of simulations, ie should be higher than
                previous
        """
        if n_simulation > self.n_simulation:
            n_simulation_old = self.n_simulation
            #transfer forecasts from old file to new file
            memmap_path_old = self.memmap_path
            self.load_memmap("r")
            forecast_array_old = self.forecast_array
            self.n_simulation = n_simulation
            self.make_memmap_path()
            self.load_memmap("w+")
            self.forecast_array[0:n_simulation_old] = forecast_array_old[:]
            del forecast_array_old
            #simulate more forecasts
            self.simulate_forecasts(range(n_simulation_old, self.n_simulation))
            if path.exists(memmap_path_old):
                os.remove(memmap_path_old)

    def load_memmap(self, mode):
        """Load the memmap file fore forecast_array

        Args:
            mode: how to read the memmap file, eg "w+", "r+", "r"
        """
        self.forecast_array = np.memmap(self.memmap_path,
                                        np.float64,
                                        mode,
                                        shape=(self.n_simulation, self.n_time))

    def del_memmap(self):
        """Delete the file handling
        """
        del self.forecast_array
        self.forecast_array = None

    def get_forecast(self):
        """Calculate statistics over all the provided forecasts
        """
        self.forecast = np.mean(self.forecast_array, 0)
        sigma_array = range(-3,4)
        forecast_quantile = np.quantile(self.forecast_array,
                                        stats.norm.cdf(sigma_array),
                                        0)
        for i in range(len(sigma_array)):
            self.forecast_sigma[sigma_array[i]] = forecast_quantile[i]
        self.forecast_median = self.forecast_sigma[0]

    def get_prob_rain(self, rainfall):
        """Get the probability if it will rain at least of a certian amount

        Args:
            rainfall: scalar, amount of rain to evaluate the probability

        Return:
            vector, a probability for each day
        """
        p_rain = np.mean(self.forecast_array > rainfall, 0)
        return p_rain

    def plot_roc_curve(self, rain_warning, rain_true):
        """Plot ROC curve, with area under curve as label

        Args:
            rain_warning: the amount of precipitation to be detected
            rain_true: observed precipitation, array, for each time point
        """
        if np.any(rain_true > rain_warning):
            p_rain_warning = self.get_prob_rain(rain_warning)
            (true_positive_array, false_positive_array, auc) = (
                roc.get_roc_curve(rain_warning, p_rain_warning, rain_true))
            roc.plot_roc_curve(
                true_positive_array, false_positive_array, auc, rain_warning)

    def get_error_rmse(self, true_y):
        """Return root mean square error

        Compare forecast with a given true_y using the root mean squared error

        Args:
            true_y: array of y

        Return:
            root mean square error
        """
        n = len(true_y)
        return np.sqrt(np.sum(np.square(self.forecast - true_y)) / n)

    def get_error_square_sqrt(self, true_y):
        """Return square mean sqrt error

        Compare forecast with a given true_y using the square mean sqrt error.
            This is the Tweedie deviance with p = 1.5

        Args:
            true_y: array of y

        Return:
            square mean sqrt error, can be infinite
        """
        n = len(true_y)
        error = np.zeros(n)
        is_infinite = False
        for i in range(n):
            y = true_y[i]
            y_hat = self.forecast[i]
            if y == 0:
                error[i] = math.sqrt(y_hat)
            else:
                if y_hat == 0:
                    is_infinite = True
                    break
                else:
                    sqrt_y_hat = math.sqrt(y_hat)
                    error[i] = (4 * math.pow(math.sqrt(y) - sqrt_y_hat, 2)
                        / sqrt_y_hat)
        if is_infinite:
            return math.inf
        else:
            return math.pow(np.sum(error) / n, 2)

    def __getitem__(self, index):
        #only to be used for plotting purposes
        #does not copy model fields
        slice_copy = Forecaster(self.time_series, self.memmap_dir)
        slice_copy.time_array = self.time_array[index]
        slice_copy.forecast_array = self.forecast_array[:, index]
        slice_copy.forecast = self.forecast[index]
        slice_copy.forecast_median = self.forecast_median[index]
        slice_copy.forecast_sigma = {}
        for key, forecast_sigma_i in self.forecast_sigma.items():
            slice_copy.forecast_sigma[key] = forecast_sigma_i[index]
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

    def start_forecast(self, n_simulation):
        #pass no model fields
        super().start_forecast(n_simulation)

    def get_simulated_forecast(self):
        """Return a TimeSeries object with simulated values, with z known
        """
        forecast_i = self.time_series.instantiate_forecast_self()
        forecast_i.simulate_given_z()
        return forecast_i
