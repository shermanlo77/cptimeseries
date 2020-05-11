import math
import os
from os import path

from cartopy import crs
import cycler
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import pandas.plotting
from statsmodels.tsa import stattools

import compound_poisson
from compound_poisson import roc
import dataset

RAIN_THRESHOLD_ARRAY = [0, 5, 10, 15]
RAIN_THRESHOLD_EXTREME_ARRAY = [0, 10, 20, 30]

def time_series(time_series, directory, prefix=""):

    pandas.plotting.register_matplotlib_converters()

    colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    cycle = cycler.cycler(color=[colours[0]], linewidth=[1])

    x = time_series.x
    y = time_series.y_array
    z = time_series.z_array
    n = len(time_series)
    n_model_field = time_series.n_model_field
    t = time_series.time_array
    poisson_rate_array = time_series.poisson_rate.value_array
    gamma_mean_array = time_series.gamma_mean.value_array
    gamma_dispersion_array = time_series.gamma_dispersion.value_array

    acf = stattools.acf(y, nlags=20, fft=True)
    try:
        pacf = stattools.pacf(y, nlags=20)
    except(stattools.LinAlgError):
        pacf = np.full(21, np.nan)

    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.plot(t, y)
    plt.xlabel("time")
    plt.ylabel("rainfall (mm)")
    plt.savefig(path.join(directory, prefix + "rainfall.pdf"))
    plt.close()

    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    rain_sorted = np.sort(y)
    cdf = np.asarray(range(n))
    plt.plot(rain_sorted, cdf)
    if np.any(rain_sorted == 0):
        non_zero_index = rain_sorted.nonzero()[0]
        if non_zero_index.size > 0:
            non_zero_index = rain_sorted.nonzero()[0][0] - 1
        else:
            non_zero_index = len(cdf) - 1
        plt.scatter(0, cdf[non_zero_index])
    plt.xlabel("rainfall (mm)")
    plt.ylabel("cumulative frequency")
    plt.savefig(path.join(directory, prefix + "cdf.pdf"))
    plt.close()

    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.bar(np.asarray(range(acf.size)), acf)
    plt.axhline(1/math.sqrt(n), linestyle='--', linewidth=1)
    plt.axhline(-1/math.sqrt(n), linestyle='--', linewidth=1)
    plt.xlabel("time (day)")
    plt.ylabel("autocorrelation")
    plt.savefig(path.join(directory, prefix + "acf.pdf"))
    plt.close()

    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.bar(np.asarray(range(pacf.size)), pacf)
    plt.axhline(1/math.sqrt(n), linestyle='--', linewidth=1)
    plt.axhline(-1/math.sqrt(n), linestyle='--', linewidth=1)
    plt.xlabel("time (day)")
    plt.ylabel("partial autocorrelation")
    plt.savefig(path.join(directory, prefix + "pacf.pdf"))
    plt.close()

    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.plot(t, poisson_rate_array)
    plt.xlabel("time")
    plt.ylabel("poisson rate")
    plt.savefig(path.join(directory, prefix + "poisson_rate.pdf"))
    plt.close()

    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.plot(t, gamma_mean_array)
    plt.xlabel("time")
    plt.ylabel("gamma mean (mm)")
    plt.savefig(path.join(directory, prefix + "gamma_mean.pdf"))
    plt.close()

    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.plot(t, gamma_dispersion_array)
    plt.xlabel("time")
    plt.ylabel("gamma dispersion")
    plt.savefig(path.join(directory, prefix + "gamma_dispersion.pdf"))
    plt.close()

    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.plot(t, z)
    plt.xlabel("time")
    plt.ylabel("Z")
    plt.savefig(path.join(directory, prefix + "z.pdf"))
    plt.close()

    file = open(path.join(directory, prefix + "parameter.txt"), "w")
    file.write(str(time_series))
    file.close()

def get_year_index_dir(time_array):
    year_index_dir = {} #key: years #value: index of times with that year
    for i, time in enumerate(time_array):
        year = time.year
        if not year in year_index_dir:
            year_index_dir[year] = []
        year_index_dir[year].append(i)
    #convert index into slice objects
    for year, index in year_index_dir.items():
        year_index_dir[year] = slice(index[0], index[-1]+1)
    return year_index_dir

def forecast(forecast, observed_rain, directory, prefix=""):

    forecast.load_memmap("r")

    pandas.plotting.register_matplotlib_converters()

    colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    cycle_forecast = cycler.cycler(color=[colours[1], colours[0]],
                                   linewidth=[1, 1],
                                   alpha=[1, 0.5])

    time_array = forecast.time_array
    year_index_dir = get_year_index_dir(time_array)

    for year, index in year_index_dir.items():

        forecast_sliced = forecast[index]
        observed_rain_i = observed_rain[index]

        forecast_i = forecast_sliced.forecast
        forecast_median_i = forecast_sliced.forecast_median
        forecast_lower_error = forecast_sliced.forecast_sigma[-1]
        forecast_upper_error = forecast_sliced.forecast_sigma[1]
        time_array_i = forecast_sliced.time_array

        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle_forecast)
        plt.fill_between(time_array_i,
                         forecast_lower_error,
                         forecast_upper_error,
                         alpha=0.25)
        plt.plot(time_array_i, forecast_i)
        plt.plot(time_array_i, observed_rain_i)
        plt.xlabel("time")
        plt.ylabel("precipitation (mm)")
        plt.savefig(
            path.join(directory, prefix + "_forecast_" + str(year) + ".pdf"))
        plt.close()

        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle_forecast)
        plt.fill_between(time_array_i,
                         forecast_lower_error,
                         forecast_upper_error,
                         alpha=0.25)
        plt.plot(time_array_i, forecast_median_i)
        plt.plot(time_array_i, observed_rain_i)
        plt.xlabel("time")
        plt.ylabel("precipitation (mm)")
        plt.savefig(
            path.join(directory,
                      prefix+"_forecast_median_"+str(year)+".pdf"))
        plt.close()

        plt.figure()
        plt.plot(time_array_i, forecast_i - observed_rain_i)
        plt.xlabel("time")
        plt.ylabel("residual (mm)")
        plt.savefig(
            path.join(directory, prefix + "_residual_" + str(year) + ".pdf"))
        plt.close()

        plt.figure()
        for rain in RAIN_THRESHOLD_ARRAY:
            roc_curve = forecast_sliced.get_roc_curve(rain, observed_rain_i)
            if not roc_curve is None:
                roc_curve.plot()
        plt.legend(loc="lower right")
        plt.savefig(path.join(directory, prefix + "_roc_" + str(year) + ".pdf"))
        plt.close()

        for rain in RAIN_THRESHOLD_ARRAY:
            plt.figure()
            plt.plot(time_array_i, forecast_sliced.get_prob_rain(rain))
            for i, date_i in enumerate(time_array_i):
                if observed_rain_i[i] > rain:
                    plt.axvline(x=date_i, color="r", linestyle=":")
            plt.xlabel("time")
            plt.ylabel("forecasted probability of > "+str(rain)+" mm")
            plt.savefig(
                path.join(directory,
                          prefix+"_prob_"+str(rain)+"_"+str(year)+".pdf"))
            plt.close()

    plt.figure()
    for rain in RAIN_THRESHOLD_EXTREME_ARRAY:
        roc_curve = forecast.get_roc_curve(rain, observed_rain)
        if not roc_curve is None:
            roc_curve.plot()
    plt.legend(loc="lower right")
    plt.savefig(path.join(directory, prefix + "_roc_all.pdf"))
    plt.close()

    file = open(path.join(directory, prefix + "_errors.txt"), "w")
    file.write("Deviance: ")
    file.write(str(forecast.get_error_square_sqrt(observed_rain)))
    file.write("\n")
    file.write("Rmse: ")
    file.write(str(forecast.get_error_rmse(observed_rain)))
    file.write("\n")
    file.close()

    forecast.del_memmap()

def downscale_forecast(forecast_array, test_set, directory, pool):
    forecast_array.load_memmap("r")
    forecast_map = ma.empty_like(test_set.rain)
    downscale = forecast_array.downscale

    series_dir = path.join(directory, "series_forecast")
    if not path.isdir(series_dir):
        os.mkdir(series_dir)
    map_dir = path.join(directory, "map_forecast")
    if not path.isdir(map_dir):
        os.mkdir(map_dir)

    for time_series in downscale.generate_unmask_time_series():
        lat_i = time_series.id[0]
        long_i = time_series.id[1]
        forecaster = time_series.forecaster
        forecast_map[:, lat_i, long_i] = forecaster.forecast_median

    #plot the rain
    angle_resolution = dataset.ANGLE_RESOLUTION
    longitude_grid = test_set.topography["longitude"] - angle_resolution / 2
    latitude_grid = test_set.topography["latitude"] + angle_resolution / 2

    rain_units = test_set.rain_units
    message_array = []
    for i, time in enumerate(test_set.time_array):
        title = "precipitation (" + rain_units + ") : " + str(time)
        file_path = path.join(map_dir, str(i) + ".png")
        message = PrintForecastMapMessage(forecast_map[i],
                                          latitude_grid,
                                          longitude_grid,
                                          title,
                                          file_path)
        message_array.append(message)
    pool.map(PrintForecastMapMessage.print, message_array)

    #plot the forecast for each location
    message_array = []
    for time_series_i, observed_rain_i in (
        zip(downscale.generate_unmask_time_series(),
            test_set.generate_unmask_rain())):
        message = PrintForecastSeriesMessage(
            series_dir, time_series_i, observed_rain_i)
        message_array.append(message)
    pool.map(PrintForecastSeriesMessage.print, message_array)

    year_index_dir = get_year_index_dir(forecast_array.time_array)
    for year, index in year_index_dir.items():
        plt.figure()
        roc_curve_array = forecast_array.get_roc_curve_array(
            RAIN_THRESHOLD_ARRAY, test_set, index)
        for roc_curve in roc_curve_array:
            roc_curve.plot()
        plt.legend(loc="lower right")
        plt.savefig(path.join(directory, "test_roc_"+str(year)+".pdf"))
        plt.close()

    plt.figure()
    roc_curve_array = forecast_array.get_roc_curve_array(
        RAIN_THRESHOLD_EXTREME_ARRAY, test_set)
    for roc_curve in roc_curve_array:
        roc_curve.plot()
    plt.legend(loc="lower right")
    plt.savefig(path.join(directory, "test_roc_full.pdf"))
    plt.close()

class PrintForecastMapMessage(object):

    def __init__(self,
                 forecast_i,
                 latitude_grid,
                 longitude_grid,
                 title,
                 file_path):
        self.forecast_i = forecast_i
        self.latitude_grid = latitude_grid
        self.longitude_grid = longitude_grid
        self.title = title
        self.file_path = file_path

    def print(self):
        self.forecast_i.mask[self.forecast_i == 0] = True

        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(self.longitude_grid,
                       self.latitude_grid,
                       self.forecast_i,
                       vmin=0,
                       vmax=15)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title(self.title)
        plt.savefig(self.file_path)
        plt.close()

class PrintForecastSeriesMessage(object):

    def __init__(self, series_dir, time_series_i, observed_rain_i):
        self.series_sub_dir =  path.join(series_dir, str(time_series_i.id))
        self.forecaster = time_series_i.forecaster
        self.observed_rain_i = observed_rain_i
        if not path.exists(self.series_sub_dir):
            os.mkdir(self.series_sub_dir)

    def print(self):
        forecast(
            self.forecaster, self.observed_rain_i, self.series_sub_dir, "test")
