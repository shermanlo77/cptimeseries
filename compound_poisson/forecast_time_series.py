import datetime
import math
import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
            forecast_i = self.instantiate_forecast()
            forecast_i.simulate()
            self.forecast_array[i] = forecast_i.y_array
        self.time_array = forecast_i.time_array
        self.get_forecast()
        self.del_memmap()

    def instantiate_forecast(self):
        return self.time_series.instantiate_forecast(self.model_field)

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

class SelfForecaster(Forecaster):

    def __init__(self, time_series, memmap_dir):
        super().__init__(time_series, memmap_dir)

    def start_forecast(self, n_simulation):
        super().start_forecast(n_simulation)

    def instantiate_forecast(self):
        return self.time_series.instantiate_forecast_self()
