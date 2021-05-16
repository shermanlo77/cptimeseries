"""Plot figures and tables for comparing CP-MCMC with ERA5 for Wales

Plot figures and tables for comparing CP-MCMC with ERA5 for Wales. The first
    point of contact is the procedure plot(). The remaining procedures are for
    plotting figures for loss, empirical distributions and spatial/temporal
    correlations.

Plots:
    -bias loss for each year (loss vs year), plots both CP-MCMC and ERA5
        on the same figure)
    -bias loss for each location as a spatial map, a figure for CP-MCMC and
        ERA5 with the same colour bar scale
    -forecast vs observed as a histogram, a figure for CP-MCMC and ERA5 with
        the same colour bar scale and same histogram binnings
    -comparing the empirical distributions with the observed using survival
        plot, a figure for CP-MCMC and ERA5
    -comparing the empirical distributions with the observed using pp
        plot, a figure for CP-MCMC and ERA5
    -comparing the empirical distributions with the observed using qq
        plot, a figure for CP-MCMC and ERA5
    -spatial correlation as a spatial map with different temporal lag, a
        figure for CP-MCMC, ERA5 and observed and they all share the same
        colour bar
    -ljung_box_pierce statistics (as a spatial map for different lags),
        comparing spatial correlation of CP-MCMC with ERA5
    -ljung_box_pierce statistics (as a histogram, averaging over space, for
        different lags), comparing spatial correlation of CP-MCMC with ERA5

Tables:
    -bias loss for the total years and seasons
"""

import math
import os
from os import path

import cycler
from cartopy import crs
import joblib
from matplotlib import pyplot as plt
import numpy as np
from numpy import ma
import pandas as pd
import pandas.plotting
import scipy

import compound_poisson
from compound_poisson.forecast import loss_segmentation
from compound_poisson.forecast import residual_analysis
from compound_poisson.forecast import time_segmentation
import dataset

LINESTYLE = ['-', '--', '-.', ':']
MONOCHROME = (cycler.cycler('color', ['k'])
              * cycler.cycler('linestyle', LINESTYLE))


def plot(model_name):
    """
    Args:
        model_name: the name of the class to compare with, eg "MultiSeries"
            would load the file "result/MultiSeries.gz"
    """

    plt.rcParams.update({'font.size': 14})
    pandas.plotting.register_matplotlib_converters()

    name_array = ["CP-MCMC (5)", "IFS"]  # labels for each downscale

    # load the data, fitted Downscale and ERA 5

    observed_data = dataset.WalesTest()

    downscale = joblib.load(path.join("result", model_name+".gz"))
    downscale.forecaster.load_memmap("r")

    era5_data = dataset.Era5Wales()
    era5 = compound_poisson.era5.Downscale(era5_data)
    era5.fit(era5_data, observed_data)

    # where to save the figures
    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)
    directory = path.join(directory, model_name)
    if not path.isdir(directory):
        os.mkdir(directory)
    directory = path.join(directory, "compare")
    if not path.isdir(directory):
        os.mkdir(directory)

    # plot comparisions
    loss_yearly(downscale, era5, name_array, directory)
    loss_seasonal(downscale, era5, name_array, directory)
    loss_spatial(downscale, era5, name_array, directory)
    compare_forecast(downscale, era5, name_array, directory)
    distribution(downscale, era5, name_array, directory)
    spatial_correlation(downscale, era5, name_array, directory)

    downscale.forecaster.del_memmap()


def loss_yearly(downscale, era5, name_array, path_figure):
    """Plot the loss for each year (loss vs time)

    Args:
        downscale: MultiSeries object
        era5: Era5 object
        name_array: array of String, length 2
        path_figure: where to save the figures
    """
    time_array = downscale.forecaster.time_array
    downscale_array = [downscale, era5]
    # yearly plot of the bias losses
    time_segmentator = time_segmentation.YearSegmentator(time_array)
    loss_segmentator_array = []
    for downscale in downscale_array:
        loss_segmentator_i = loss_segmentation.Downscale(downscale.forecaster)
        loss_segmentator_i.evaluate_loss(time_segmentator)
        loss_segmentator_array.append(loss_segmentator_i)

    for i_loss, Loss in enumerate(loss_segmentation.LOSS_CLASSES):

        # array of arrays, one for each time_series in time_series_array
        # for each array, contains array of loss for each time point
        bias_loss_plot_array = []
        bias_median_loss_plot_array = []

        for downscale_i, loss_segmentator_i in zip(
                downscale_array, loss_segmentator_array):
            bias_loss_plot, bias_median_loss_plot = (
                loss_segmentator_i.get_bias_plot(i_loss))
            bias_loss_plot_array.append(bias_loss_plot)
            bias_median_loss_plot_array.append(bias_median_loss_plot)

        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(MONOCHROME)
        for downscale_label, bias_plot_array in zip(name_array,
                                                    bias_loss_plot_array):
            plt.plot(loss_segmentator_i.time_array,
                     bias_plot_array,
                     label=downscale_label)
        plt.legend()
        plt.ylabel(Loss.get_axis_bias_label())
        plt.xlabel("year")
        plt.xticks(rotation=45)
        plt.savefig(
            path.join(path_figure, Loss.get_short_bias_name()+"_mean.pdf"),
            bbox_inches="tight")
        plt.close()

        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(MONOCHROME)
        for downscale_label, bias_plot_array in zip(
                    name_array, bias_median_loss_plot_array):
            plt.plot(loss_segmentator_i.time_array,
                     bias_plot_array,
                     label=downscale_label)
        plt.legend()
        plt.ylabel(Loss.get_axis_bias_label())
        plt.xlabel("year")
        plt.xticks(rotation=45)
        plt.savefig(
            path.join(path_figure, Loss.get_short_bias_name()+"_median.pdf"),
            bbox_inches="tight")
        plt.close()


def loss_seasonal(downscale, era5, name_array, path_figure):
    """Plot the loss for each season (as a table)

    Args:
        downscale: MultiSeries object
        era5: Era5 object
        name_array: array of String, length 2
        path_figure: where to save the figures
    """
    time_array = downscale.forecaster.time_array
    downscale_array = [downscale, era5]

    # plot table of test set bias loss
    time_segmentator_array = {
        "all_years": time_segmentation.AllInclusive(time_array),
        "spring": time_segmentation.SpringSegmentator(time_array),
        "summer": time_segmentation.SummerSegmentator(time_array),
        "autumn": time_segmentation.AutumnSegmentator(time_array),
        "winter": time_segmentation.WinterSegmentator(time_array),
    }
    time_segmentator_names = list(time_segmentator_array.keys())

    # array of loss_segmentator objects, for each time series
    # dim 0: for each time series
    # dim 1: for each time segmentator
    loss_array = []

    # plot the table (for mean, the median bias)
    for i, downscale_i in enumerate(downscale_array):
        loss_array.append([])
        for time_segmentator_k in time_segmentator_array.values():
            forecaster_i = downscale_i.forecaster
            loss_i = loss_segmentation.Downscale(forecaster_i)
            loss_i.evaluate_loss(time_segmentator_k)
            loss_array[i].append(loss_i)

    for i_loss, Loss in enumerate(loss_segmentation.LOSS_CLASSES):

        # using training set size 5 years to get bootstrap variance, this is
        # used to guide the number of decimial places to use
        n_decimial = 3
        float_format = ("{:."+str(n_decimial)+"f}").format

        # table of losses
        # columns: for each time segmentator
        # rows: for each time series
        loss_mean_array = []
        loss_median_array = []

        # plot the table (for mean, the median bias)
        for i_downscale, downscale_i in enumerate(downscale_array):
            loss_mean_array.append([])
            loss_median_array.append([])
            for loss_segmentator_i in loss_array[i_downscale]:
                loss = loss_segmentator_i.loss_all_array[i_loss]
                loss_mean_array[i_downscale].append(loss.get_bias_loss())
                loss_median_array[i_downscale].append(
                    loss.get_bias_median_loss())

        for prefix, loss_table in zip(
                ["mean", "median"], [loss_mean_array, loss_median_array]):
            data_frame = pd.DataFrame(
                loss_table, name_array, time_segmentator_names)
            path_to_table = path.join(
                path_figure, prefix+"_"+Loss.get_short_bias_name()+".txt")
            data_frame.to_latex(path_to_table,
                                float_format=float_format)


def loss_spatial(downscale, era5, name_array, path_figure):
    """Plot the loss (mean absolute error) (as a spatial map)

    Args:
        downscale: MultiSeries object
        era5: Era5 object
        name_array: array of String, length 2
        path_figure: where to save the figures
    """
    observed_data = downscale.forecaster.data
    downscale_array = [downscale, era5]

    loss_map_array = []  # store loss as a map, array of numpy matrix
    loss_min = math.inf  # for ensuring the colour bar is the same
    loss_max = 0  # for ensuring the colour bar is the same
    for downscale in downscale_array:
        loss_map = ma.empty_like(observed_data.rain[0])
        for forecaster_i, observed_rain_i in (
                zip(downscale.forecaster.generate_forecaster_no_memmap(),
                    observed_data.generate_unmask_rain())):
            lat_i = forecaster_i.time_series.id[0]
            long_i = forecaster_i.time_series.id[1]

            loss_i = compound_poisson.forecast.loss.MeanAbsoluteError(
                downscale.forecaster.n_simulation)
            loss_i.add_data(forecaster_i, observed_rain_i)
            loss_bias_i = loss_i.get_bias_median_loss()
            loss_map[lat_i, long_i] = loss_bias_i

            if loss_bias_i < loss_min:
                loss_min = loss_bias_i
            if loss_bias_i > loss_max:
                loss_max = loss_bias_i

        loss_map_array.append(loss_map)

    angle_resolution = dataset.ANGLE_RESOLUTION
    longitude_grid = (
        observed_data.topography["longitude"] - angle_resolution / 2)
    latitude_grid = (
        observed_data.topography["latitude"] + angle_resolution / 2)

    # plot the losses
    for loss_map, downscale_name in zip(loss_map_array, name_array):
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid,
                       latitude_grid,
                       loss_map,
                       vmin=loss_min,
                       vmax=loss_max,
                       cmap='Greys')
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.savefig(
            path.join(
                path_figure, downscale_name+"_mae_map.pdf"),
            bbox_inches="tight")
        plt.close()


def compare_forecast(downscale, era5, name_array, path_figure):
    """forecast vs observed as a histogram, a figure for CP-MCMC and ERA5 with
        the same colour bar scale and same histogram binnings

    Args:
        downscale: MultiSeries object
        era5: Era5 object
        name_array: array of String, length 2
        path_figure: where to save the figures
    """
    downscale_array = [downscale, era5]
    for i, downscale_i in enumerate(downscale_array):
        residual_plot = residual_analysis.ResidualLnqqPlotter()

        residual_plot.add_downscale(downscale_i.forecaster)

        # plot residual data
        residual_plot.plot_heatmap([[0, 4.3], [0, 4.3]], 1.8, 7.2, 'Greys')
        plt.savefig(
            path.join(path_figure,
                      name_array[i]+"_residual_qq_hist.pdf"),
            bbox_inches="tight")
        plt.close()


def distribution(downscale, era5, name_array, path_figure):
    """Compare the distribtuions

    Compare the distribtuions of CP-MCMC with observed and ERA5 with observed.
        Survival plot, pp plot and qq plot

    Args:
        downscale: MultiSeries object
        era5: Era5 object
        name_array: array of String, length 2
        path_figure: where to save the figures
    """

    downscale_comparer = downscale.forecaster.compare_dist_with_observed()
    era5_comparer = era5.forecaster.compare_dist_with_observed()

    # survival plot
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(MONOCHROME)
    downscale_comparer.plot_survival_forecast(name_array[0])
    era5_comparer.plot_survival_forecast(name_array[1])
    era5_comparer.plot_survival_observed("observed")
    downscale_comparer.adjust_survival_plot()
    plt.legend()
    plt.savefig(path.join(path_figure, "survival.pdf"), bbox_inches="tight")
    plt.close()

    # pp plot
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(MONOCHROME)
    downscale_comparer.plot_pp(name_array[0])
    era5_comparer.plot_pp(name_array[1])
    downscale_comparer.adjust_pp_plot()
    plt.legend()
    plt.savefig(path.join(path_figure, "pp.pdf"), bbox_inches="tight")
    plt.close()

    # qq plot
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(MONOCHROME)
    downscale_comparer.plot_qq(name_array[0])
    downscale_comparer.adjust_qq_plot()
    era5_comparer.plot_qq(name_array[1])
    plt.legend()
    plt.savefig(path.join(path_figure, "qq.pdf"), bbox_inches="tight")
    plt.close()


def spatial_correlation(downscale, era5, name_array, path_figure):
    """Plot spatial correlations

    Plot spatial correlation as a spatial map with different temporal lag, a
        figure for CP-MCMC, ERA5 and observed and they all share the same
        colour bar.
    Spatial correlation is compared to the centre of mass for Wales (can be
        referred to as the reference)
    The median over all forecasts is taken first, followed by spatial
        correlation. ie, shown are the spatial correlation of the median
        forecast, not the median spatial correlation of the forecasts
    Plot ljung_box_pierce statistics (as a spatial map for different lags),
        comparing spatial correlation of CP-MCMC with ERA5
    Plot ljung_box_pierce statistics (as a histogram, averaging over space, for
        different lags), comparing spatial correlation of CP-MCMC with ERA5

    Args:
        downscale: MultiSeries object
        era5: Era5 object
        name_array: array of String, length 2
        path_figure: where to save the figures
    """

    test_set = downscale.forecaster.data
    time_length = len(test_set)

    angle_resolution = dataset.ANGLE_RESOLUTION
    longitude_grid = test_set.topography["longitude"] - angle_resolution / 2
    latitude_grid = test_set.topography["latitude"] + angle_resolution / 2

    reference = [10, 17]  # index for the centre of mass for Wales

    # time series at the centre of mass for cp-mcmc, era 5 and observed
    forecast_reference = None
    era5_reference = None
    test_set_reference = test_set.rain[:, reference[0], reference[1]]
    # retrive the cp-mcmc forecast and era 5
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

    # array of spatial map of spatial correlation, one for each time lag
    forecast_cross_correlation_array = []
    era5_cross_correlation_array = []
    test_set_cross_correlation_array = []
    n_lag = 10

    # for each lag, calculate and plot spatial correlation
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

        vmax = ma.max([forecast_cross_correlation.max(),
                       era5_cross_correlation.max(),
                       test_set_cross_correlation.max(),
                       ])
        vmin = ma.min([forecast_cross_correlation.min(),
                       era5_cross_correlation.min(),
                       test_set_cross_correlation.min(),
                       ])

        cross_correlation_array = [
            forecast_cross_correlation,
            era5_cross_correlation,
            test_set_cross_correlation,
        ]
        label_array = ["forecast", "era5", "observed"]
        for cross_correlation, label in zip(
                cross_correlation_array, label_array):

            plt.rcParams.update({'font.size': 18})
            plt.figure()
            ax = plt.axes(projection=crs.PlateCarree())
            im = ax.pcolor(longitude_grid,
                           latitude_grid,
                           cross_correlation,
                           cmap='Greys',
                           vmin=vmin,
                           vmax=vmax)
            plt.hlines(latitude_grid[reference[0],
                       reference[1]] - angle_resolution / 2,
                       longitude_grid.min(),
                       longitude_grid.max(),
                       colors='k',
                       linestyles='dashed')
            plt.vlines(longitude_grid[reference[0],
                       reference[1]] + angle_resolution / 2,
                       latitude_grid.min(),
                       latitude_grid.max(),
                       colors='k',
                       linestyles='dashed')
            ax.coastlines(resolution="50m")
            plt.colorbar(im)
            ax.set_aspect("auto", adjustable=None)
            plt.savefig(
                path.join(path_figure,
                          "correlation_"+label+"_"+str(i_lag)+".pdf"),
                bbox_inches="tight")
            plt.close()

    # ljung_box_pierce statistics, for comparing cp-mcmc with observed

    forecast_cross_correlation_array = ma.asarray(
        forecast_cross_correlation_array)

    test_set_cross_correlation_array = ma.asarray(
        test_set_cross_correlation_array)

    # array of numpy array, for each lag
    # each numpy array contains the lbp statistic for each location
    lbp_array = []

    for i_lag in range(n_lag):
        forecast_statistic = ljung_box_pierce(
            forecast_cross_correlation_array, time_length, i_lag)
        test_set_statistic = ljung_box_pierce(
            test_set_cross_correlation_array, time_length, i_lag)

        f_statistic = ma.log(forecast_statistic) - ma.log(test_set_statistic)
        f_statistic = ma.exp(f_statistic)

        # spatial plot
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid,
                       latitude_grid,
                       f_statistic,
                       cmap='Greys')
        plt.hlines(latitude_grid[reference[0],
                   reference[1]] - angle_resolution / 2,
                   longitude_grid.min(),
                   longitude_grid.max(),
                   colors='k',
                   linestyles='dashed')
        plt.vlines(longitude_grid[reference[0],
                   reference[1]] + angle_resolution / 2,
                   latitude_grid.min(),
                   latitude_grid.max(),
                   colors='k',
                   linestyles='dashed')
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.savefig(
            path.join(path_figure, "correlation_lbp_"+str(i_lag)+".pdf"),
            bbox_inches="tight")
        plt.close()

        f_statistic = f_statistic[np.logical_not(f_statistic.mask)]
        f_statistic = f_statistic.data
        lbp_array.append(f_statistic.flatten())

    # histogram of statistics
    plt.figure()
    plt.boxplot(lbp_array, positions=range(n_lag))
    plt.xlabel("lag")
    plt.ylabel("ratio of Ljung-Box")
    plt.savefig(
        path.join(path_figure, "correlation_lbp_ratio.pdf"),
        bbox_inches="tight")


def ljung_box_pierce(cross_correlation_array, length, n_lag):
    """Calculate Ljung-Box-Pierce statistics

    Args:
        cross_correlation_array: array of spatial maps of correlations, for
            each lag
        length: length of the time series
        n_lag: integer, maximum temporal lag to sum

    Return:
        2d array, same shape as value_map[0, :, :], contains lbp statistics
    """
    statistic = ma.empty_like(cross_correlation_array[0])
    statistic[np.logical_not(statistic.mask)] = 0
    for i in range(n_lag+1):
        statistic += (ma.exp(2*ma.log(cross_correlation_array[i])
                      - math.log(length - i)))
    return statistic


def spatial_cross_correlation(reference_series, value_map, lag=0):
    """Correlation of one location with every other location

    Args:
        reference_series: the time series of a location to be comapred with
        value_map: dim 0: time, dim 1 and 2: 2d map, values of each location
            and time
        lag: amount of temporal lag

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
