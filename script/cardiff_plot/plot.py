"""Figures and tables for Cardiff for different size training sets

Plots AUC for different precipitation and different size training sets
    x-axis: precipitation
    y-axos: AUC
    different lines for different size training sets

Table of bias loss
    Columns: RMSE, R10, MAE, MAE10
    Rows: Different size training sets and ERA5
Tables for:
    the entire test set, spring, summer, autumn, winter
"""

import math
import os
from os import path

import cycler
import joblib
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
import pandas.plotting

import compound_poisson
from compound_poisson.forecast import loss_segmentation
from compound_poisson.forecast import residual_analysis
from compound_poisson.forecast import time_segmentation
import dataset

LINESTYLE = ['-', '--', '-.', ':']
LINESTYLE2 = ['--', '-.', '-', ':']

def main():

    monochrome = (cycler.cycler('color', ['k'])
        * cycler.cycler('linestyle', LINESTYLE))
    monochrome2 = (cycler.cycler('color', ['k'])
        * cycler.cycler('linestyle', LINESTYLE2))
    plt.rcParams.update({'font.size': 14})

    #where to save the figures
    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)

    seed = random.SeedSequence(301608752619507842997952162996242447135)
    rng = random.RandomState(random.MT19937(seed))

    era5 = compound_poisson.era5.TimeSeries()
    era5.fit(dataset.Era5Cardiff())

    observed_data = dataset.CardiffTest()
    observed_rain = observed_data.rain
    time_array = observed_data.time_array

    training_size_array = [1, 5]
    script_dir_array = [
        "cardiff_1_20",
        "cardiff_5_20",
    ]
    for i, dir_i in enumerate(script_dir_array):
        script_dir_array[i] = path.join("..", dir_i)

    time_series_name_array = [] #time series for each training set
    time_series_array = []
    #will need to update the location of each time series memmap_path because
        #they would be using relative paths
    for i, dir_i in enumerate(script_dir_array):
        time_series = joblib.load(
            path.join(dir_i, "result", "TimeSeriesHyperSlice.gz"))
        old_dir = time_series.forecaster.memmap_path
        time_series.forecaster.memmap_path = path.join(dir_i, old_dir)
        time_series.forecaster.load_memmap("r")
        time_series_array.append(time_series)
        time_series_name_array.append(
            "CP-MCMC ("+str(training_size_array[i])+")")

    #plot auc for varying precipitation

    #array of array:
        #for each training set, then for each value in rain_array
    auc_array = []
    bootstrap_error_array = []
    n_bootstrap = 32
    rain_array = [0, 5, 10, 15]
    for i_training_size, size_i in enumerate(training_size_array):
        auc_array.append([])
        bootstrap_error_array.append([])
        forecaster_i = time_series_array[i_training_size].forecaster
        for rain_i in rain_array:
            roc_i = forecaster_i.get_roc_curve(rain_i, observed_rain)
            auc_array[i_training_size].append(roc_i.area_under_curve)

            bootstrap_i_array = []
            for j_bootstrap in range(n_bootstrap):
                bootstrap = forecaster_i.bootstrap(rng)
                roc_ij = bootstrap.get_roc_curve(rain_i, observed_rain)
                bootstrap_i_array.append(
                    math.pow(
                        roc_ij.area_under_curve - roc_i.area_under_curve, 2))
            bootstrap_error_array[i_training_size].append(
                math.sqrt(np.mean(bootstrap_i_array)))

    #figure format
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(monochrome)
    for i_training_size, size_i in enumerate(training_size_array):
        plt.plot(rain_array,
                 auc_array[i_training_size],
                 label=time_series_name_array[i_training_size])
    plt.ylim([0.5, 1])
    plt.xlabel("precipitation (mm)")
    plt.ylabel("Area under ROC curve")
    plt.legend()
    plt.savefig(path.join(directory, "auc.pdf"), bbox_inches="tight")
    plt.close()

    #table format
    rain_label_array = []
    for rain in rain_array:
        rain_label_array.append(str(rain)+" mm")
    #table format with uncertainity values
    auc_table = []
    for auc_i, error_i in zip(auc_array, bootstrap_error_array):
        auc_table.append([])
        for auc_ij, error_ij in zip(auc_i, error_i):
            auc_table[-1].append(
                "${:0.4f}\pm {:0.4f}$".format(auc_ij, error_ij))

    data_frame = pd.DataFrame(
        np.asarray(auc_table).T, rain_label_array, time_series_name_array)
    data_frame.to_latex(path.join(directory, "auc.txt"), escape=False)

    #add era5 (for loss evaluation)
    #roc unavailable for era5
    time_series_array.append(era5)
    time_series_name_array.append("IFS")

    #yearly plot of the bias losses
    time_segmentator = time_segmentation.YearSegmentator(time_array)
    loss_segmentator_array = []
    for time_series_i in time_series_array:
        loss_segmentator_i = loss_segmentation.TimeSeries(
            time_series_i.forecaster, observed_rain)
        loss_segmentator_i.evaluate_loss(time_segmentator)
        loss_segmentator_array.append(loss_segmentator_i)

    pandas.plotting.register_matplotlib_converters()
    for i_loss, Loss in enumerate(loss_segmentation.LOSS_CLASSES):

        #array of arrays, one for each time_series in time_series_array
            #for each array, contains array of loss for each time point
        bias_loss_plot_array = []
        bias_median_loss_plot_array = []

        for time_series_i, loss_segmentator_i in zip(
            time_series_array, loss_segmentator_array):
            bias_loss_plot, bias_median_loss_plot = (
                loss_segmentator_i.get_bias_plot(i_loss))
            bias_loss_plot_array.append(bias_loss_plot)
            bias_median_loss_plot_array.append(bias_median_loss_plot)

        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(monochrome2)
        for time_series_label, bias_plot_array in zip(time_series_name_array,
            bias_loss_plot_array):
            plt.plot(loss_segmentator_i.time_array,
                     bias_plot_array,
                     label=time_series_label)
        plt.legend()
        plt.ylabel(Loss.get_axis_bias_label())
        plt.xlabel("year")
        plt.xticks(rotation=45)
        plt.savefig(
            path.join(directory, Loss.get_short_bias_name()+"_mean.pdf"),
            bbox_inches="tight")
        plt.close()

        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(monochrome2)
        for time_series_label, bias_plot_array in zip(time_series_name_array,
            bias_median_loss_plot_array):
            plt.plot(loss_segmentator_i.time_array,
                     bias_plot_array,
                     label=time_series_label)
        plt.legend()
        plt.ylabel(Loss.get_axis_bias_label())
        plt.xlabel("year")
        plt.xticks(rotation=45)
        plt.savefig(
            path.join(directory, Loss.get_short_bias_name()+"_median.pdf"),
            bbox_inches="tight")
        plt.close()

    #plot table of test set bias loss
    time_segmentator_array = {
        "all_years": time_segmentation.AllInclusive(time_array),
        "spring": time_segmentation.SpringSegmentator(time_array),
        "summer": time_segmentation.SummerSegmentator(time_array),
        "autumn": time_segmentation.AutumnSegmentator(time_array),
        "winter": time_segmentation.WinterSegmentator(time_array),
    }
    loss_name_array = []
    float_format_array = [] #each column to have a certain decimial values
    for Loss in loss_segmentation.LOSS_CLASSES:
        #using training set size 5 years to get bootstrap variance, this is used
            #to guide the number of decimial places to use
        loss_name_array.append(Loss.get_short_bias_name())
        n_decimial = number_of_decimial_places(
            time_series_array[1], observed_rain, Loss, 100, rng)
        float_format_array.append(("{:."+str(n_decimial)+"f}").format)

    #plot the table (for mean, the median bias)
    for time_key, time_segmentator_k in time_segmentator_array.items():
        #table of losses
            #columns: for each loss
            #rows: for each time series
        loss_array = []
        loss_median_array = []
        for i, time_series_i in enumerate(time_series_array):
            loss_array.append([])
            loss_median_array.append([])
            forecaster_i = time_series_i.forecaster
            loss_i = loss_segmentation.TimeSeries(forecaster_i, observed_rain)
            loss_i.evaluate_loss(time_segmentator_k)
            for loss_ij in loss_i.loss_all_array:
                loss_array[i].append(loss_ij.get_bias_loss())
                loss_median_array[i].append(loss_ij.get_bias_median_loss())

        for prefix, loss_table in zip(
            ["mean", "median"], [loss_array, loss_median_array]):
            data_frame = pd.DataFrame(
                loss_table, time_series_name_array, loss_name_array)
            path_to_table = path.join(directory, prefix+"_"+time_key+".txt")
            data_frame.to_latex(path_to_table,
                                formatters=float_format_array)

    for i, time_series_i in enumerate(time_series_array):
        residual_plot = residual_analysis.ResidualLnqqPlotter()

        #add residuals data
        residual_plot.add_data(time_series_i.forecaster, observed_rain)

        #plot residual data
        residual_plot.plot_heatmap([[0, 3.8], [0, 3.8]], 1.8, 5.3, 'Greys')
        plt.savefig(
            path.join(directory,
                      time_series_name_array[i]+"_residual_qq_hist.pdf"),
            bbox_inches="tight")
        plt.close()

    for time_series_i in time_series_array:
        time_series_i.forecaster.del_memmap()

def number_of_decimial_places(
    time_series, observed_rain, Loss, n_bootstrap, rng):
    """Return the recommend number of decimial places by using the variance of
        the bootstrapped forecats

    Only suitable if the bootstrap standard deviation is in the order of
        decimial places

    Args:
        time_series: the forecast to be bootstrapped
        observed_rain: array of observed precipitation
        Loss: class for a loss
        n_bootstrap: number of bootstrap samples
        rng:

    Return:
        integer, number of decimial places
    """
    loss = Loss(time_series.forecaster.n_simulation)
    forecaster = time_series.forecaster
    loss_array = []
    #bootstrap
    for i in range(n_bootstrap):
        bootstrap = forecaster.bootstrap(rng)
        loss.add_data(bootstrap, observed_rain)
        loss_array.append(loss.get_bias_loss())
    loss_std = np.std(loss_array, ddof=1)
    #round to the nearest magnitiude
    return -round(math.log10(loss_std))

if __name__ == "__main__":
    main()
