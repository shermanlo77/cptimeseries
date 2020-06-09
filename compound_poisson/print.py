"""Functions for printing TimeSeries, Downscale, Forecast objects.

These functions could be made to methods for the corresponding classes...
"""

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
RAIN_THRESHOLD_EXTREME_ARRAY = [0, 5, 10, 15]

def time_series(time_series, directory, prefix=""):
    """Print figures for a given TimeSeries object

    Save figure for precipitation time series
    Save figure for cumulative frequency of precipitation in the time series
    Save figures for sample autocorrelation and partial autocorrelation
    Save figures for the poisson rate, gamma mean, gamma dispersion, latent
        variable Z over time
    Save the TimeSeries in text (converted to string which shows parameters)

    Args:
        time_series: compound_poisson.TimeSeries object
        directory: where to save the figures
        prefix: what to name the figures
    """

    #required when plotting times on an axis
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

    #get autocorrelations
    acf = stattools.acf(y, nlags=20, fft=True)
    try:
        pacf = stattools.pacf(y, nlags=20)
    except(stattools.LinAlgError):
        pacf = np.full(21, np.nan)

    #print precipitation time series
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.plot(t, y)
    plt.xlabel("time")
    plt.ylabel("rainfall (mm)")
    plt.savefig(path.join(directory, prefix + "rainfall.pdf"))
    plt.close()

    #print precipitation cumulative frequency
    #draw dot for mass at 0 mm
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

    #plot sample autocorrelation
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

    #plot sample partial autocorrelation
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

    #plot the poisson rate over time
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.plot(t, poisson_rate_array)
    plt.xlabel("time")
    plt.ylabel("poisson rate")
    plt.savefig(path.join(directory, prefix + "poisson_rate.pdf"))
    plt.close()

    #plot the gamma mean over time
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.plot(t, gamma_mean_array)
    plt.xlabel("time")
    plt.ylabel("gamma mean (mm)")
    plt.savefig(path.join(directory, prefix + "gamma_mean.pdf"))
    plt.close()

    #plot the gamme dispersion over time
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.plot(t, gamma_dispersion_array)
    plt.xlabel("time")
    plt.ylabel("gamma dispersion")
    plt.savefig(path.join(directory, prefix + "gamma_dispersion.pdf"))
    plt.close()

    #plot the latent variable z over time
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(cycle)
    plt.plot(t, z)
    plt.xlabel("time")
    plt.ylabel("Z")
    plt.savefig(path.join(directory, prefix + "z.pdf"))
    plt.close()

    #print the parameters in text
    file = open(path.join(directory, prefix + "parameter.txt"), "w")
    file.write(str(time_series))
    file.close()

def get_year_index_dir(time_array):
    """For splitting the time_array into different years, used for plotting
        figures for different years

    Args:
        time_array: array of dates

    Return:
        dictionary, keys are years, values are slice objects pointing to that
            year
    """
    #key: years #value: index of times with that year
    year_index_dir = {}
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
    """For plotting compound_poisson.forecast.time_series.Forecaster objects

    Plots the forecasts for each year:
        -forecast (mean) with 68% credible interval vs observe
        -forecast (median) with 68% credible interval vs observe
        -residuals
        -ROC curve
        -forecast probability of more than some amount (see
            RAIN_THRESHOLD_ARRAY) of precipitation
    Plots:
        -area under ROC curve for each year
        -root mean square error for each year

    Args:
        forecast: Forecaster object
        observed_rain: numpy array of observed precipitation
        directory: where to save the figure
        prefix: what to name the figure
    """

    #required to load forecast samples
    forecast.load_memmap("r")

    #required for plotting dates
    pandas.plotting.register_matplotlib_converters()

    #setting plotting properties (for both observed and prediction)
    colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    cycle_forecast = cycler.cycler(color=[colours[1], colours[0]],
                                   linewidth=[1, 1],
                                   alpha=[1, 0.5])

    #split time_array into different years
    time_array = forecast.time_array
    year_index_dir = get_year_index_dir(time_array)
    year_array = np.asarray(list(year_index_dir))

    rmse_array = [] #array for storing rmse for each year
    #dictionary of AUC for different thresholds, key is amount of rain, value is
        #array containing AUC for each year
    auc_array = {}
    for rain in RAIN_THRESHOLD_ARRAY:
        auc_array[rain] = []

    #for each year
    for year, index in year_index_dir.items():

        #slice the current forecast and observation and plot
        forecast_sliced = forecast[index]
        observed_rain_i = observed_rain[index]
        rmse_array.append(forecast_sliced.get_error_rmse(observed_rain_i))

        forecast_i = forecast_sliced.forecast
        forecast_median_i = forecast_sliced.forecast_median
        forecast_lower_error = forecast_sliced.forecast_sigma[-1]
        forecast_upper_error = forecast_sliced.forecast_sigma[1]
        time_array_i = forecast_sliced.time_array

        #plot forecast and observation time series
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

        #plot forecast using median instead
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

        #plot residual
        plt.figure()
        plt.plot(time_array_i, forecast_i - observed_rain_i)
        plt.xlabel("time")
        plt.ylabel("residual (mm)")
        plt.savefig(
            path.join(directory, prefix + "_residual_" + str(year) + ".pdf"))
        plt.close()

        #plot ROC curve, save AUC as well
        plt.figure()
        for rain in RAIN_THRESHOLD_ARRAY:
            roc_curve = forecast_sliced.get_roc_curve(rain, observed_rain_i)
            if not roc_curve is None:
                roc_curve.plot()
                auc = roc_curve.area_under_curve
            else:
                auc = np.nan
            auc_array[rain].append(auc)
        plt.legend(loc="lower right")
        plt.savefig(path.join(directory, prefix + "_roc_" + str(year) + ".pdf"))
        plt.close()

        #plot probability of more than rain precipitation
        #plot vertical red line to show when that even actually happened
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

    #plot auc for each year
    plt.figure()
    for rain in RAIN_THRESHOLD_ARRAY:
        label = str(rain) + " mm"
        auc = np.asarray(auc_array[rain])
        is_number = np.logical_not(np.isnan(auc))
        plt.plot(year_array[is_number], auc[is_number], '-o', label=label)
    plt.legend()
    plt.xlabel("year")
    plt.ylabel("area under curve")
    plt.savefig(path.join(directory, prefix + "_auc.pdf"))
    plt.close()

    #plot rmse for each year
    plt.figure()
    plt.plot(year_array, rmse_array, '-o')
    plt.xlabel("year")
    plt.ylabel("root mean square error (mm)")
    plt.savefig(path.join(directory, prefix + "_rmse.pdf"))
    plt.close()

    #plot roc curve for the entire test set
    plt.figure()
    for rain in RAIN_THRESHOLD_EXTREME_ARRAY:
        roc_curve = forecast.get_roc_curve(rain, observed_rain)
        if not roc_curve is None:
            roc_curve.plot()
    plt.legend(loc="lower right")
    plt.savefig(path.join(directory, prefix + "_roc_all.pdf"))
    plt.close()

    #memmap of forecasts no longer needed
    forecast.del_memmap()

def downscale_forecast(forecast_array, test_set, directory, pool):
    """For plotting compound_poisson.forecast.downscale.Forecaster objects

    Plots:
        -plots the spatial forecast map for each time point (in parallel)
        -plots rmse spatial map
        -plots forecast time series for each spatial point (in parallel)
        -for each year:
            -plot roc curve (use entire area)
        -plots auc for each year
        -plots rmse for each year
        -plots roc curve using entire test set

    Args:
        forecast_array: compound_poisson.forecast.downscale.Forecaster object
        test_set: Data object
        directory: where to save the figures
        pool: multiprocess object to do tasks in parallel
    """

    #required to load memmap containing forecasts
    forecast_array.load_memmap("r")

    downscale = forecast_array.downscale

    angle_resolution = dataset.ANGLE_RESOLUTION
    longitude_grid = test_set.topography["longitude"] - angle_resolution / 2
    latitude_grid = test_set.topography["latitude"] + angle_resolution / 2
    rain_units = test_set.rain_units

    #forecast map, 3 dimensions, same as test set rain, prediction of
        #precipitation for each point in space and time, 0th dimension is time,
        #remaining is space
    forecast_map = ma.empty_like(test_set.rain)
    #root mean square error, 2 dimension, for each point in space
    rmse_map = ma.empty_like(test_set.rain[0])
    #area under curve, dictionary, keys: for each precipitation in
        #RAIN_THRESHOLD_ARRAY, value: array, auc for each year
    auc_array = {}
    #root mean square error for each year
    rmse_array = []
    for rain in RAIN_THRESHOLD_ARRAY:
        auc_array[rain] = []

    year_index_dir = get_year_index_dir(forecast_array.time_array)
    year_array = np.asarray(list(year_index_dir))

    series_dir = path.join(directory, "series_forecast")
    if not path.isdir(series_dir):
        os.mkdir(series_dir)
    map_dir = path.join(directory, "map_forecast")
    if not path.isdir(map_dir):
        os.mkdir(map_dir)

    #get forecast (median) and rmse for the maps
    for time_series_i, observed_rain_i in (
        zip(downscale.generate_unmask_time_series(),
            test_set.generate_unmask_rain())):
        lat_i = time_series_i.id[0]
        long_i = time_series_i.id[1]
        forecaster = time_series_i.forecaster
        forecast_map[:, lat_i, long_i] = forecaster.forecast_median
        rmse_map[lat_i, long_i] = forecaster.get_error_rmse(observed_rain_i)

    #plot the spatial forecast for each time (in parallel)
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

    #plot rmse
    plt.figure()
    ax = plt.axes(projection=crs.PlateCarree())
    im = ax.pcolor(longitude_grid,
                   latitude_grid,
                   rmse_map)
    ax.coastlines(resolution="50m")
    plt.colorbar(im)
    ax.set_aspect("auto", adjustable=None)
    plt.savefig(path.join(directory, "rmse_map.pdf"))
    plt.close()

    #plot the forecast (time series) for each location (in parallel)
    message_array = []
    for time_series_i, observed_rain_i in (
        zip(downscale.generate_unmask_time_series(),
            test_set.generate_unmask_rain())):
        message = PrintForecastSeriesMessage(
            series_dir, time_series_i, observed_rain_i)
        message_array.append(message)
    pool.map(PrintForecastSeriesMessage.print, message_array)

    #for each year, plot roc curve, get rmse
    for (year, index) in year_index_dir.items():
        plt.figure()
        roc_curve_array = forecast_array.get_roc_curve_array(
            RAIN_THRESHOLD_ARRAY, test_set, index)
        for i_rain, roc_curve in enumerate(roc_curve_array):
            if not roc_curve is None:
                roc_curve.plot()
                auc = roc_curve.area_under_curve
            else:
                auc = np.nan
            auc_array[RAIN_THRESHOLD_ARRAY[i_rain]].append(auc)
        plt.legend(loc="lower right")
        plt.savefig(path.join(directory, "test_roc_"+str(year)+".pdf"))
        plt.close()

        #work out root mean square over space
        rmse_array_i = []
        for time_series_i, observed_rain_i in (
            zip(downscale.generate_unmask_time_series(),
                test_set.generate_unmask_rain())):
            forecaster_i = time_series_i.forecaster
            forecaster_i.load_memmap("r")
            forecast_sliced_i = time_series_i.forecaster[index]
            observed_rain_sliced_i = observed_rain_i[index]
            rmse_array_i.append(forecast_sliced_i.get_error_rmse(
                observed_rain_sliced_i))
            forecaster_i.del_memmap()
        #root mean square, so need to sum the squares
        rmse_array.append(math.sqrt(np.mean(np.square(rmse_array_i))))

    #plot auc for each year
    plt.figure()
    for rain in RAIN_THRESHOLD_ARRAY:
        label = str(rain) + " " + rain_units
        auc = np.asarray(auc_array[rain])
        is_number = np.logical_not(np.isnan(auc))
        plt.plot(year_array[is_number], auc[is_number], '-o', label=label)
    plt.legend()
    plt.xlabel("year")
    plt.ylabel("area under curve")
    plt.savefig(path.join(directory, "auc.pdf"))
    plt.close()

    #plot rmse for each year
    plt.figure()
    plt.plot(year_array, rmse_array, '-o')
    plt.xlabel("year")
    plt.ylabel("root mean square error (" + rain_units + ")")
    plt.savefig(path.join(directory, "rmse.pdf"))
    plt.close()

    #plot roc curve over the entire test set
    plt.figure()
    roc_curve_array = forecast_array.get_roc_curve_array(
        RAIN_THRESHOLD_EXTREME_ARRAY, test_set)
    for roc_curve in roc_curve_array:
        roc_curve.plot()
    plt.legend(loc="lower right")
    plt.savefig(path.join(directory, "test_roc_full.pdf"))
    plt.close()

class PrintForecastMapMessage(object):
    """For printing forecast over space for a given time point (in parallel)
    """

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
    """For printing forecast time series for a given spatial point (in parallel)
    """

    def __init__(self, series_dir, time_series_i, observed_rain_i):
        self.series_sub_dir =  path.join(series_dir, str(time_series_i.id))
        self.forecaster = time_series_i.forecaster
        self.observed_rain_i = observed_rain_i
        if not path.exists(self.series_sub_dir):
            os.mkdir(self.series_sub_dir)

    def print(self):
        forecast(
            self.forecaster, self.observed_rain_i, self.series_sub_dir, "test")
