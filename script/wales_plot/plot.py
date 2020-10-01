"""Figures and tables for Isle of Man
"""

import math
import os
from os import path

import joblib
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
import pandas.plotting

import compound_poisson
from compound_poisson.forecast import loss_segmentation
from compound_poisson.forecast import time_segmentation
import dataset

def main():

    #where to save the figures
    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)

    observed_data = dataset.WalesTest()
    time_array = observed_data.time_array

    downscale_name_array = [] #time series for each training set
    downscale_array = []

    dir_i = path.join("..", "wales_5_20")
    downscale = joblib.load(
        path.join(dir_i, "result", "Downscale.gz"))
    old_dir = downscale.forecaster.memmap_path
    downscale.forecaster.memmap_path = path.join(dir_i, old_dir)

    for time_series in downscale.generate_unmask_time_series():
        forecaster = time_series.forecaster
        old_dir = forecaster.memmap_path
        forecaster.memmap_path = path.join(dir_i, old_dir)

    downscale.forecaster.load_memmap("r")
    downscale.forecaster.load_locations_memmap("r")
    downscale_array.append(downscale)
    downscale_name_array.append("Compound-Poisson")

    era5 = dataset.Era5IsleOfMan()
    downscale = compound_poisson.era5.Downscale(era5)
    downscale.fit(era5, observed_data)
    downscale_array.append(downscale)
    downscale_name_array.append("ERA5")

    #yearly plot of the bias losses
    time_segmentator = time_segmentation.YearSegmentator(time_array)
    loss_segmentator_array = []
    for downscale in downscale_array:
        print(downscale)
        loss_segmentator_i = loss_segmentation.Downscale()
        loss_segmentator_i.evaluate_loss(
            downscale.forecaster, time_segmentator)
        loss_segmentator_array.append(loss_segmentator_i)

    pandas.plotting.register_matplotlib_converters()
    for i_loss, Loss in enumerate(loss_segmentation.LOSS_CLASSES):

        #array of arrays, one for each time_series in time_series_array
            #for each array, contains array of loss for each time point
        bias_loss_plot_array = []
        bias_median_loss_plot_array = []

        for downscale_i, loss_segmentator_i in zip(
            downscale_array, loss_segmentator_array):
            bias_loss_plot, bias_median_loss_plot = (
                loss_segmentator_i.get_bias_plot(i_loss))
            bias_loss_plot_array.append(bias_loss_plot)
            bias_median_loss_plot_array.append(bias_median_loss_plot)

        plt.figure()
        for downscale_label, bias_plot_array in zip(downscale_name_array,
            bias_loss_plot_array):
            plt.plot(loss_segmentator_i.time_array,
                     bias_plot_array,
                     label=downscale_label)
        plt.legend()
        plt.ylabel(Loss.get_axis_bias_label())
        plt.xlabel("year")
        plt.savefig(
            path.join(directory, Loss.get_short_bias_name()+" _mean.pdf"))
        plt.close()

        plt.figure()
        for downscale_label, bias_plot_array in zip(downscale_name_array,
            bias_median_loss_plot_array):
            plt.plot(loss_segmentator_i.time_array,
                     bias_plot_array,
                     label=downscale_label)
        plt.legend()
        plt.ylabel(Loss.get_axis_bias_label())
        plt.xlabel("year")
        plt.savefig(
            path.join(directory, Loss.get_short_bias_name()+" _median.pdf"))
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
        if Loss is compound_poisson.forecast.loss.MeanAbsoluteError:
            n_decimial = 4
        else:
            n_decimial = 3
        float_format_array.append(("{:."+str(n_decimial)+"f}").format)

    #plot the table (for mean, the median bias)
    for time_key, time_segmentator_k in time_segmentator_array.items():
        #table of losses
            #columns: for each loss
            #rows: for each time series
        loss_array = []
        loss_median_array = []
        for i, downscale_i in enumerate(downscale_array):
            loss_array.append([])
            loss_median_array.append([])
            forecaster_i = downscale_i.forecaster
            loss_i = loss_segmentation.Downscale()
            loss_i.evaluate_loss(forecaster_i, time_segmentator_k)
            for loss_ij in loss_i.loss_all_array:
                loss_array[i].append(loss_ij.get_bias_loss())
                loss_median_array[i].append(loss_ij.get_bias_median_loss())

        for prefix, loss_table in zip(
            ["mean", "median"], [loss_array, loss_median_array]):
            data_frame = pd.DataFrame(
                loss_table, downscale_name_array, loss_name_array)
            path_to_table = path.join(directory, prefix+"_"+time_key+".txt")
            data_frame.to_latex(path_to_table,
                                formatters=float_format_array)

    for downscale_i in downscale_array:
        downscale.forecaster.del_memmap()
        downscale.forecaster.del_locations_memmap()

if __name__ == "__main__":
    main()
