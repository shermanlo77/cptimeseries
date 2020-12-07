import math
import os
from os import path

from cartopy import crs
import cycler
import joblib
from matplotlib import pyplot as plt
import numpy as np
from numpy import ma
import scipy
from scipy import interpolate
from scipy.ndimage import filters

import compound_poisson
import dataset

LINESTYLE = ['-', '--', '-.', ':']

def main():

    monochrome = (cycler.cycler('color', ['k'])
        * cycler.cycler('linestyle', LINESTYLE))
    plt.rcParams.update({'font.size': 14})

    #where to save the figures
    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)

    test_set = dataset.WalesTest()
    time_length = len(test_set)

    angle_resolution = dataset.ANGLE_RESOLUTION
    longitude_grid = test_set.topography["longitude"] - angle_resolution / 2
    latitude_grid = test_set.topography["latitude"] + angle_resolution / 2

    dir = path.join("..", "wales")
    downscale = joblib.load(
        path.join(dir, "result", "MultiSeries.gz"))

    era5_data = dataset.Era5Wales()
    era5 = compound_poisson.era5.Downscale(era5_data)
    era5.fit(era5_data, test_set)

    downscale_name = "CP-MCMC (5)"
    era5_name = "IFS"
    observed_name = "observed"
    old_dir = downscale.forecaster.memmap_path
    downscale.forecaster.memmap_path = path.join(dir, old_dir)

    for forecaster_i in downscale.forecaster.generate_forecaster_no_memmap():
        old_dir = forecaster_i.memmap_path
        forecaster_i.memmap_path = path.join(dir, old_dir)

    reference = [10, 17]

    #forecast map, 3 dimensions, same as test_set.rain
        #prediction of precipitation for each point in space and time
        #0th dimension is time, remaining is space
    forecast_reference = None
    era5_reference = None
    forecast_array = ma.empty_like(test_set.rain)
    era5_array = ma.empty_like(test_set.rain)
    for time_series_i in downscale.generate_unmask_time_series():
        forecast_array[:, time_series_i.id[0], time_series_i.id[1]] = (
            time_series_i.forecaster.forecast_median)
        if time_series_i.id == reference:
            forecast_reference = time_series_i.forecaster.forecast_median
    for time_series_i in era5.generate_unmask_time_series():
        era5_array[:, time_series_i.id[0], time_series_i.id[1]] = (
            time_series_i.forecaster.forecast_median)
        if time_series_i.id == reference:
            era5_reference = time_series_i.forecaster.forecast_median
    test_set_reference = test_set.rain[:, reference[0], reference[1]]

    forecast_cross_correlation_array = []
    era5_cross_correlation_array = []
    test_set_cross_correlation_array = []
    n_lag = 10

    for i_lag in range(n_lag):
        forecast_cross_correlation = spatial_cross_correlation(
            forecast_reference, forecast_array, i_lag)
        forecast_cross_correlation_array.append(forecast_cross_correlation)

        era5_cross_correlation = spatial_cross_correlation(
            era5_reference, era5_array, i_lag)
        era5_cross_correlation_array.append(era5_cross_correlation)

        test_set_cross_correlation = spatial_cross_correlation(
            test_set_reference, test_set.rain, i_lag)
        test_set_cross_correlation_array.append(test_set_cross_correlation)

        vmax = ma.max(
            [forecast_cross_correlation.max(),
            era5_cross_correlation.max(),
            test_set_cross_correlation.max(),
            ])
        vmin = ma.min(
            [forecast_cross_correlation.min(),
            era5_cross_correlation.min(),
            test_set_cross_correlation.min(),
            ])

        plt.rcParams.update({'font.size': 18})
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid,
                       latitude_grid,
                       forecast_cross_correlation,
                       cmap='Greys',
                       vmin=vmin,
                       vmax=vmax)
        plt.hlines(latitude_grid[reference[0], reference[1]] - angle_resolution / 2,
                   longitude_grid.min(),
                   longitude_grid.max(),
                   colors='k',
                   linestyles='dashed')
        plt.vlines(longitude_grid[reference[0], reference[1]] + angle_resolution / 2,
                   latitude_grid.min(),
                   latitude_grid.max(),
                   colors='k',
                   linestyles='dashed')
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.savefig(
            path.join(directory, "correlation_forecast_"+str(i_lag)+".pdf"),
            bbox_inches="tight")
        plt.close()

        plt.rcParams.update({'font.size': 18})
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid,
                       latitude_grid,
                       era5_cross_correlation,
                       cmap='Greys',
                       vmin=vmin,
                       vmax=vmax)
        plt.hlines(latitude_grid[reference[0], reference[1]] - angle_resolution / 2,
                   longitude_grid.min(),
                   longitude_grid.max(),
                   colors='k',
                   linestyles='dashed')
        plt.vlines(longitude_grid[reference[0], reference[1]] + angle_resolution / 2,
                   latitude_grid.min(),
                   latitude_grid.max(),
                   colors='k',
                   linestyles='dashed')
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.savefig(
            path.join(directory, "correlation_era5_"+str(i_lag)+".pdf"),
            bbox_inches="tight")
        plt.close()

        plt.rcParams.update({'font.size': 18})
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid,
                       latitude_grid,
                       test_set_cross_correlation,
                       cmap='Greys',
                       vmin=vmin,
                       vmax=vmax)
        plt.hlines(latitude_grid[reference[0], reference[1]] - angle_resolution / 2,
                   longitude_grid.min(),
                   longitude_grid.max(),
                   colors='k',
                   linestyles='dashed')
        plt.vlines(longitude_grid[reference[0], reference[1]] + angle_resolution / 2,
                   latitude_grid.min(),
                   latitude_grid.max(),
                   colors='k',
                   linestyles='dashed')
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.savefig(
            path.join(directory, "correlation_observed_"+str(i_lag)+".pdf"),
            bbox_inches="tight")
        plt.close()

    forecast_cross_correlation_array = ma.asarray(
        forecast_cross_correlation_array)

    test_set_cross_correlation_array = ma.asarray(
        test_set_cross_correlation_array)

    lbp_array = []

    for i_lag in range(n_lag):
        forecast_statistic = ljung_box_pierce(
            forecast_cross_correlation_array, time_length, i_lag)
        test_set_statistic = ljung_box_pierce(
            test_set_cross_correlation_array, time_length, i_lag)

        f_statistic = ma.log(forecast_statistic) - ma.log(test_set_statistic)
        f_statistic = ma.exp(f_statistic)

        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid,
                       latitude_grid,
                       f_statistic,
                       cmap='Greys')
        plt.hlines(latitude_grid[reference[0], reference[1]] - angle_resolution / 2,
                   longitude_grid.min(),
                   longitude_grid.max(),
                   colors='k',
                   linestyles='dashed')
        plt.vlines(longitude_grid[reference[0], reference[1]] + angle_resolution / 2,
                   latitude_grid.min(),
                   latitude_grid.max(),
                   colors='k',
                   linestyles='dashed')
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.savefig(
            path.join(directory, "correlation_lbp_"+str(i_lag)+".pdf"),
            bbox_inches="tight")
        plt.close()

        f_statistic = f_statistic[np.logical_not(f_statistic.mask)]
        f_statistic = f_statistic.data
        lbp_array.append(f_statistic.flatten())

    plt.figure()
    plt.boxplot(lbp_array, positions=range(n_lag))
    plt.xlabel("lag")
    plt.ylabel("ratio of Ljung-Box")
    plt.savefig(
        path.join(directory, "correlation_lbp_ratio.pdf"),
        bbox_inches="tight")

def ljung_box_pierce(cross_correlation_array, length, n_lag):
    statistic = ma.empty_like(cross_correlation_array[0])
    statistic[np.logical_not(statistic.mask)] = 0
    for i in range(n_lag+1):
        statistic += ma.exp(2*ma.log(cross_correlation_array[i]) - math.log(length - i))
    return statistic

def spatial_cross_correlation(reference_series, value_map, lag=0):
    """Correlation of one location with every other location

    Args:
        reference_series: the time series of a location to be comapred with
        value_map: dim 0: time, dim 1 and 2: 2d map, values of each location and
            time

    Return:
        2d array, same shape as value_map[0, :, :], contains correlations
    """
    cross_correlation = value_map[0].copy()
    length = len(reference_series)
    if type(cross_correlation) is ma.core.MaskedArray:
        mask = cross_correlation.mask
    else:
        mask = np.zeros(cross_correlation.shape, dtype=np.bool_)
    for i in range(cross_correlation.shape[0]):
        for j in range(cross_correlation.shape[1]):
            if not mask[i, j]:
                cross_correlation[i, j] = scipy.stats.pearsonr(
                    reference_series[0:(length-lag)],
                    value_map[lag:length, i, j])[0]
    return cross_correlation

if __name__ == "__main__":
    main()
