"""Figures and tables for comparing CP-MCMC with ERA5 for Wales

Plots:
    -bias loss for each year
    -bias loss for each location (with CP-MCMC and ERA5 to have the same colour
        bar scale)
    -log residuals plots (with CP-MCMC and ERA5 to have the same colour
        bar scale and same histogram binnings)

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

import compound_poisson
from compound_poisson.forecast import loss_segmentation
from compound_poisson.forecast import residual_analysis
from compound_poisson.forecast import time_segmentation
import dataset

LINESTYLE = ['-', '--', '-.', ':']

def main():

    #where to save the figures
    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)
    monochrome = (cycler.cycler('color', ['k'])
        * cycler.cycler('linestyle', LINESTYLE))
    plt.rcParams.update({'font.size': 14})
    pandas.plotting.register_matplotlib_converters()

    observed_data = dataset.WalesTest()
    time_array = observed_data.time_array

    downscale_name_array = [] #time series for each training set
    downscale_array = []

    dir_i = path.join("..", "wales_5_20")
    downscale = joblib.load(
        path.join(dir_i, "result", "Downscale.gz"))
    old_dir = downscale.forecaster.memmap_path
    downscale.forecaster.memmap_path = path.join(dir_i, old_dir)

    for forecaster in downscale.forecaster.generate_forecaster_no_memmap():
        old_dir = forecaster.memmap_path
        forecaster.memmap_path = path.join(dir_i, old_dir)

    downscale.forecaster.load_memmap("r")
    downscale_array.append(downscale)
    downscale_name_array.append("CP-MCMC (5)")

    era5 = dataset.Era5Wales()
    downscale = compound_poisson.era5.Downscale(era5)
    downscale.fit(era5, observed_data)
    downscale_array.append(downscale)
    downscale_name_array.append("IFS")

    #yearly plot of the bias losses
    time_segmentator = time_segmentation.YearSegmentator(time_array)
    loss_segmentator_array = []
    for downscale in downscale_array:
        loss_segmentator_i = loss_segmentation.Downscale(downscale.forecaster)
        loss_segmentator_i.evaluate_loss(time_segmentator)
        loss_segmentator_array.append(loss_segmentator_i)

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
        ax = plt.gca()
        ax.set_prop_cycle(monochrome)
        for downscale_label, bias_plot_array in zip(downscale_name_array,
            bias_loss_plot_array):
            plt.plot(loss_segmentator_i.time_array,
                     bias_plot_array,
                     label=downscale_label)
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
        ax.set_prop_cycle(monochrome)
        for downscale_label, bias_plot_array in zip(downscale_name_array,
            bias_median_loss_plot_array):
            plt.plot(loss_segmentator_i.time_array,
                     bias_plot_array,
                     label=downscale_label)
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
            loss_i = loss_segmentation.Downscale(forecaster_i)
            loss_i.evaluate_loss(time_segmentator_k)
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

    loss_map_array = []
    loss_min = math.inf
    loss_max = 0
    for downscale in downscale_array:
        loss_map = ma.empty_like(observed_data.rain[0])
        for forecaster_i, observed_rain_i in (
            zip(downscale.forecaster.generate_forecaster_no_memmap(),
                observed_data.generate_unmask_rain())):
            lat_i = forecaster_i.time_series.id[0]
            long_i = forecaster_i.time_series.id[1]

            loss_i = compound_poisson.forecast.loss.MeanAbsoluteError(
                forecaster.n_simulation)
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

    #plot the losses
    for loss_map, downscale_name in zip(loss_map_array, downscale_name_array):
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
                directory, downscale_name+"_mae_map.pdf"),
            bbox_inches="tight")
        plt.close()

    for i, downscale_i in enumerate(downscale_array):
        residual_plot = residual_analysis.ResidualLnqqPlotter()

        residual_plot.add_downscale(downscale_i.forecaster)

        #plot residual data
        residual_plot.plot_heatmap([[0, 4.3], [0, 4.3]], 1.8, 7.2, 'Greys')
        plt.savefig(
            path.join(directory,
                      downscale_name_array[i]+"_residual_qq_hist.pdf"),
            bbox_inches="tight")
        plt.close()

    for downscale_i in downscale_array:
        downscale.forecaster.del_memmap()

if __name__ == "__main__":
    main()
