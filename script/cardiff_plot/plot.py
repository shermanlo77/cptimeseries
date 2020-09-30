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

import joblib
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import pandas as pd

import compound_poisson
from compound_poisson.forecast import loss_segmentation
from compound_poisson.forecast import time_segmentation
import dataset

def main():

    #where to save the figures
    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)

    era5 = compound_poisson.era5.TimeSeries()
    era5.fit(dataset.Era5Cardiff())

    observed_data = dataset.CardiffTest()
    observed_rain = observed_data.rain
    time_array = observed_data.time_array

    training_size_array = [1, 5, 10, 20]
    script_dir_array = [
        "cardiff_1_20",
        "cardiff_5_20",
        "cardiff_10_20",
        "cardiff_20_20"
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
        time_series_name_array.append(str(training_size_array[i]) + " year(s)")

    #array of array:
        #for each training set, then for each value in rain_array
    auc_array = []
    rain_array = [0, 5, 10, 15, 20, 25, 30]
    for i_training_size, size_i in enumerate(training_size_array):
        auc_array.append([])
        forecaster_i = time_series_array[i_training_size].forecaster
        for rain_i in rain_array:
            roc_i = forecaster_i.get_roc_curve(rain_i, observed_rain)
            auc_array[i_training_size].append(roc_i.area_under_curve)

    plt.figure()
    for i_training_size, size_i in enumerate(training_size_array):
        plt.plot(rain_array, auc_array[i_training_size],
                 label=time_series_name_array[i_training_size])
    plt.yscale('log')
    plt.ylim([0.5, 1])
    plt.xlabel("precipitation (mm)")
    plt.ylabel("Area under ROC curve")
    plt.legend()
    plt.savefig(path.join(directory, "auc.pdf"))
    plt.close()

    #add era5 (for loss evaluation)
    #roc unavailable for era5
    time_series_array.append(era5)
    time_series_name_array.append("ERA5")

    time_segmentator_array = {
        "all_years": time_segmentation.AllInclusive(time_array),
        "spring": time_segmentation.SpringSegmentator(time_array),
        "summer": time_segmentation.SummerSegmentator(time_array),
        "autumn": time_segmentation.AutumnSegmentator(time_array),
        "winter": time_segmentation.WinterSegmentator(time_array),
    }
    loss_name_array = []
    float_format_array = [] #each column to have a certain decimial values
    seed = random.SeedSequence(301608752619507842997952162996242447135)
    rng = random.RandomState(random.MT19937(seed))
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
            loss_i = loss_segmentation.TimeSeries()
            loss_i.evaluate_loss(
                forecaster_i, observed_rain, time_segmentator_k)
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
