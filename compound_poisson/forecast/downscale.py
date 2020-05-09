import datetime
import math
import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from compound_poisson.forecast import time_series

class Forecaster(time_series.Forecaster):

    def __init__(self, downscale, memmap_dir):
        self.downscale = downscale
        self.time_array = None
        self.data = None
        self.forecast_array = None
        self.n_time = None
        self.n_simulation = 0
        self.memmap_dir = memmap_dir
        self.memmap_path = None

    def start_forecast(self, n_simulation, data):
        self.data = data
        self.n_time = len(data)
        self.n_simulation = n_simulation
        self.time_array = data.time_array
        self.make_memmap_path()
        self.load_memmap("w+")
        self.simulate_forecasts(range(n_simulation))
        self.del_memmap()

    def resume_forecast(self, n_simulation):
        if n_simulation > self.n_simulation:
            n_simulation_old = self.n_simulation
            #transfer forecasts from old file to new file
            memmap_path_old = self.memmap_path
            self.load_memmap("r")
            forecast_array_old = self.forecast_array
            self.n_simulation = n_simulation
            self.make_memmap_path()
            self.load_memmap("w+")
            for i in range(len(self.forecast_array)):
                self.forecast_array[i, 0:n_simulation_old] = (
                    forecast_array_old[i])
            del forecast_array_old
            #simulate more forecasts
            self.simulate_forecasts(range(n_simulation_old, self.n_simulation))
            if path.exists(memmap_path_old):
                os.remove(memmap_path_old)

    def make_memmap_path(self):
        datetime_id = str(datetime.datetime.now())
        datetime_id = datetime_id.replace("-", "")
        datetime_id = datetime_id.replace(":", "")
        datetime_id = datetime_id.replace(" ", "")
        datetime_id = datetime_id[0:14]
        file_name = ("_" + type(self).__name__ + type(self.downscale).__name__
            + datetime_id + ".dat")
        self.memmap_path = path.join(self.memmap_dir, file_name)

    def simulate_forecasts(self, index_range):
        area_unmask = self.downscale.area_unmask
        n_total_parameter = self.downscale.n_total_parameter

        forecast_message = []
        for i_space, time_series in enumerate(
            self.downscale.generate_unmask_time_series()):
            #extract model fields for each unmasked time_series
            lat_i = time_series.id[0]
            long_i = time_series.id[1]
            x_i = self.data.get_model_field(lat_i, long_i)
            #extract mcmc chain corresponding to this location
            parameter_mcmc = self.downscale.parameter_mcmc[:]
            parameter_mcmc = parameter_mcmc[
                :, range(i_space, n_total_parameter, area_unmask)]
            time_series.parameter_mcmc = parameter_mcmc
            time_series.burn_in = self.downscale.burn_in

            message = ForecastMessage(
                time_series, x_i, self.n_simulation, self.memmap_path, i_space)
            forecast_message.append(message)

        time_series_array = self.downscale.pool.map(
            ForecastMessage.forecast, forecast_message)
        self.downscale.replace_unmask_time_series(time_series_array)

    def load_memmap(self, mode):
        self.forecast_array = np.memmap(self.memmap_path,
                                        np.float64,
                                        mode,
                                        shape=(self.downscale.area_unmask,
                                               self.n_simulation,
                                               self.n_time))

    def generate_time_series_forecaster(self):
        for time_series in self.downscale.generate_unmask_time_series():
            forecaster = time_series.forecaster
            forecaster.load_memmap("r")
            yield time_series.forecaster

class TimeSeriesForecaster(time_series.Forecaster):

    def __init__(self, time_series, memmap_path, i_space):
        super().__init__(time_series, path.dirname(memmap_path))
        self.i_space = i_space
        self.memmap_path = memmap_path

    def start_forecast(self, n_simulation, model_field):
        self.model_field = model_field
        self.n_time = len(model_field)
        self.n_simulation = n_simulation
        self.load_memmap("r+")
        self.simulate_forecasts(range(n_simulation))

    def resume_forecast(self, n_simulation):
        if n_simulation > self.n_simulation:
            n_simulation_old = self.n_simulation
            self.n_simulation = n_simulation
            self.load_memmap("r+")
            self.simulate_forecasts(range(n_simulation_old, self.n_simulation))
            self.del_memmap()

    def load_memmap(self, mode):
        self.forecast_array = np.memmap(self.memmap_path, np.float64, mode)
        self.forecast_array = np.reshape(
            self.forecast_array, (-1, self.n_simulation, self.n_time))
        self.forecast_array = self.forecast_array[self.i_space]

class ForecastMessage(object):
    #message to pass, see Downscale.forecast

    def __init__(
        self, time_series, model_field, n_simulation, memmap_path, i_space):
        self.time_series = time_series
        self.model_field = model_field
        self.n_simulation = n_simulation
        self.memmap_path = memmap_path
        self.i_space = i_space

    def forecast(self):
        self.time_series.forecast(self.model_field, self.n_simulation,
            self.memmap_path, self.i_space)
        return self.time_series
