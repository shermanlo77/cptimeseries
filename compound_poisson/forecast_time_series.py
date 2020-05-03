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

    Contain Monte Carlo forecasts in forecast_array as an array of TimeSeries
        objects. Add forecasts using the method append(). Call the method
        get_forecast() to calculate statistcs of all forecats. The statistics
        are stored as member variables.
    Used by the methods TimeSeries.forecast() and TimeSeries.forecast_self().

    Attributes:
        forecast_array: memmap of TimeSeries objects
        forecast: expectation of all forecasts
        forecast_error: std of all forecasts
        forecast_median: median of all forecasts
        forecast_sigma: dictoary of z sigma errors of all forecasts
        time_array: array, containing time stamps for each point in the forecast
        n: length of time series
        n_simulation: length of forecast_array
        memmap_path
    """

    def __init__(self, time_series, memmap_dir):
        self.time_series = time_series
        self.time_array = None
        self.model_field = None
        self.forecast_array = None
        self.forecast = None
        self.forecast_error = None
        self.forecast_median = None
        self.forecast_sigma = {}
        self.time_array = None
        self.n_time = None
        self.n_simulation = 0
        self.memmap_dir = memmap_dir
        self.memmap_path = None

    def make_memmap_path(self):
        datetime_id = str(datetime.datetime.now())
        datetime_id = datetime_id.replace("-", "")
        datetime_id = datetime_id.replace(":", "")
        datetime_id = datetime_id.replace(" ", "")
        datetime_id = datetime_id[0:14]
        file_name = ("_" + type(self).__name__ + type(self.time_series).__name__
            + datetime_id + ".dat")
        self.memmap_path = path.join(self.memmap_dir, file_name)

    def start_forecast(self, n_simulation, model_field=None):
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
        for i in index_range:
            print("Predictive sample", i)
            forecast_i = self.get_simulated_forecast()
            self.forecast_array[i] = forecast_i.y_array
        self.time_array = forecast_i.time_array
        self.get_forecast()
        self.del_memmap()

    def get_simulated_forecast(self):
        forecast_i = self.time_series.instantiate_forecast(self.model_field)
        forecast_i.simulate()
        return forecast_i

    def resume_forecast(self, n_simulation):
        if n_simulation > self.n_simulation:
            n_simulation_old = self.n_simulation
            memmap_path_old = self.memmap_path
            self.load_memmap("r")
            forecast_array_old = self.forecast_array
            self.n_simulation = n_simulation
            self.make_memmap_path()
            self.load_memmap("w+")
            self.forecast_array[0:n_simulation_old] = forecast_array_old[:]
            del forecast_array_old
            self.simulate_forecasts(range(n_simulation_old, self.n_simulation))
            if path.exists(memmap_path_old):
                os.remove(memmap_path_old)

    def load_memmap(self, mode):
        self.forecast_array = np.memmap(self.memmap_path,
                                        np.float64,
                                        mode,
                                        shape=(self.n_simulation, self.n_time))

    def del_memmap(self):
        del self.forecast_array
        self.forecast_array = None

    def get_forecast(self):
        """Calculate statistics over all the provided forecasts
        """
        self.forecast = np.mean(self.forecast_array, 0)
        self.forecast_error = np.std(self.forecast_array, 0, ddof=1)
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
        """Plot ROC curve
        """
        p_rain_warning = self.get_prob_rain(rain_warning)
        (true_positive_array, false_positive_array, auc) = roc.get_roc_curve(
            rain_warning, p_rain_warning, rain_true)
        plt.step(false_positive_array,
                 true_positive_array,
                 where="post",
                 label=str(rain_warning)+" mm, AUC = "+str(round(auc, 3)))
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")

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

class SelfForecaster(Forecaster):

    def __init__(self, time_series, memmap_dir):
        super().__init__(time_series, memmap_dir)

    def start_forecast(self, n_simulation):
        super().start_forecast(n_simulation)

    def get_simulated_forecast(self):
        forecast_i = self.time_series.instantiate_forecast_self()
        forecast_i.simulate_given_z()
        return forecast_i
