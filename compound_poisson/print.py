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
    rain_threshold_array = [0, 5, 10, 15]
    rain_threshold_extreme_array = [0, 10, 20, 30]

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
        for rain in rain_threshold_array:
            forecast_sliced.plot_roc_curve(rain, observed_rain_i)
        plt.legend()
        plt.savefig(path.join(directory, prefix + "_roc_" + str(year) + ".pdf"))
        plt.close()

        for rain in rain_threshold_array:
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
    for rain in rain_threshold_extreme_array:
        forecast.plot_roc_curve(rain, observed_rain)
    plt.legend()
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

def downscale_forecast(forecast_array, test_set, directory):
    forecast_array.load_memmap("r")
    forecast_map = ma.empty_like(test_set.rain)

    series_dir = path.join(directory, "series_forecast")
    if not path.isdir(series_dir):
        os.mkdir(series_dir)
    map_dir = path.join(directory, "map_forecast")
    if not path.isdir(map_dir):
        os.mkdir(map_dir)

    mask = test_set.rain.mask[0]
    counter = 0
    observed_rain_mask_array = []

    for lat_i in range(forecast_map.shape[1]):
        for long_i in range(forecast_map.shape[2]):
            if not mask[lat_i, long_i]:
                observed_rain_mask_array.append(
                    test_set.get_rain(lat_i, long_i))
                forecast_i = forecast_array.forecast_array[counter]
                forecast_map[:, lat_i, long_i] = np.median(forecast_i, 0)
                counter += 1
    observed_rain_mask_array = np.asarray(observed_rain_mask_array)

    #plot the rain
    longitude_grid = test_set.topography["longitude"]
    latitude_grid = test_set.topography["latitude"]
    angle_resolution = dataset.ANGLE_RESOLUTION
    for i, time in enumerate(test_set.time_array):

        if i < 365:

            rain_i = forecast_map[i]
            rain_i.mask[rain_i == 0] = True

            plt.figure()
            ax = plt.axes(projection=crs.PlateCarree())
            im = ax.pcolor(longitude_grid - angle_resolution / 2,
                           latitude_grid + angle_resolution / 2,
                           rain_i,
                           vmin=0,
                           vmax=15)
            ax.coastlines(resolution="50m")
            plt.colorbar(im)
            ax.set_aspect("auto", adjustable=None)
            plt.title(
                "precipitation (" + test_set.rain_units + ") : " + str(time))
            plt.savefig(path.join(map_dir, str(i) + ".png"))
            plt.close()

    #plot the forecast for each location
    downscale = forecast_array.downscale
    for i, time_series_i in enumerate(downscale.generate_unmask_time_series()):
        series_sub_dir = path.join(series_dir, str(time_series_i.id))
        if not path.exists(series_sub_dir):
            os.mkdir(series_sub_dir)
        forecast(time_series_i.forecaster,
                 observed_rain_mask_array[i],
                 series_sub_dir,
                 "test")

    rain_threshold_array = [0, 5, 10, 15]
    year_index_dir = get_year_index_dir(forecast_array.time_array)
    for year, index in year_index_dir.items():
        observed_rain_i = observed_rain_mask_array[:, index].flatten()
        plt.figure()
        for rain_warning in rain_threshold_array:
            prob_rain_array = []
            for forecast_i in forecast_array.generate_time_series_forecaster():
                prob_rain_array.append(
                    forecast_i[index].get_prob_rain(rain_warning))
                forecast_i.del_memmap()
            prob_rain_array = np.asarray(prob_rain_array).flatten()
            true_positive_array, false_positive_array, auc = (
                roc.get_roc_curve(
                    rain_warning, prob_rain_array, observed_rain_i))
            roc.plot_roc_curve(
                true_positive_array, false_positive_array, auc, rain_warning)
        plt.legend()
        plt.savefig(path.join(directory, "test_roc_"+str(year)+".pdf"))
        plt.close()

    rain_threshold_extreme_array = [0, 10, 20, 30]
    observed_rain_mask_array = observed_rain_mask_array.flatten()
    for rain_warning in rain_threshold_extreme_array:
        prob_rain_array = []
        if np.any(rain_warning > observed_rain_mask_array):
            for forecast_i in forecast_array.generate_time_series_forecaster():
                prob_rain_array.append(forecast_i.get_prob_rain(rain_warning))
                forecast_i.del_memmap()
            prob_rain_array = np.asarray(prob_rain_array).flatten()
            true_positive_array, false_positive_array, auc = (
                roc.get_roc_curve(
                    rain_warning, prob_rain_array,observed_rain_mask_array))
            roc.plot_roc_curve(
                true_positive_array, false_positive_array, auc, rain_warning)
    plt.legend()
    plt.savefig(path.join(directory, "test_roc_full.pdf"))
    plt.close()
