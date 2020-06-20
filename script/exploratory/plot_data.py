"""Functions for exploratory analysis

Variables looked at:
    precipitation (fine grid) over space and time
    model fields (coarse grid but also fine grid using default interpolation)
        over space and time
    topography over space (fine grid but also coarse grid using default
        interpolation)
Cities looked at:
    London
    Cardiff (exact coordinates cannot be used as the resolution used would get
        an area over water)
    Edinburgh
    Belfast
    Dublin
    and more can be hard coded
"""

import math
import multiprocessing
import os
from os import path
import pathlib

from cartopy import crs
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import pandas as pd
import scipy.stats
from statsmodels.tsa import stattools

import dataset

def plot_data(data):
    """For plotting figures given the dataset

    See the individual functions for description of figures:
        plot_rain()
        plot_model_fields()
        plot_matrix()
        plot_topography()

    Args:
        data: dataset.data object
    """

    #required so that python datetime can be converted and can be plotted on a
        #graph
    pd.plotting.register_matplotlib_converters()

    path_here = pathlib.Path(__file__).parent.absolute()
    data_name = type(data).__name__
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    figure_dir = path.join(figure_dir, data_name)
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)

    pool = multiprocessing.Pool()

    plot_rain(data, figure_dir, pool)
    plot_model_fields(data, figure_dir, pool)
    plot_matrix(data, figure_dir)
    plot_topography(data, figure_dir)

def plot_rain(data, figure_dir, pool):
    """Plot figures for the precipitation

    Plots:
        Mean precipitation over time (heatmap)
        For each city:
            precipitation time series
            cdf
            acf
            pacf
        For each time (first year for now):
            observed precipitation (heatmap)

    Args:
        data: dataset.data object
        figure_dir: where to save the figures
    """
    rain_dir = path.join(figure_dir, "rain")
    if not path.isdir(rain_dir):
        os.mkdir(rain_dir)
    map_dir = path.join(rain_dir, "map")
    if not path.isdir(map_dir):
        os.mkdir(map_dir)
    series_dir = path.join(rain_dir, "series")
    if not path.isdir(series_dir):
        os.mkdir(series_dir)

    time = data.time_array
    angle_resolution = dataset.ANGLE_RESOLUTION
    latitude_grid = data.topography["latitude"] + angle_resolution / 2
    longitude_grid = data.topography["longitude"] - angle_resolution / 2

    #get the mean rainfall (mean over time) for each point in space
    mean_rainfall = np.mean(data.rain, 0)

    #heatmap plot the mean rainfall
    plt.figure()
    ax = plt.axes(projection=crs.PlateCarree())
    im = ax.pcolor(longitude_grid, latitude_grid, mean_rainfall)
    ax.coastlines(resolution="50m")
    plt.colorbar(im)
    ax.set_aspect("auto", adjustable=None)
    plt.title("mean precipitation (" + data.rain_units + ")")
    plt.savefig(path.join(series_dir, "rainfall_mean.pdf"))
    plt.close()

    #plot the rainfall as a time series for each city
    for city in dataset.CITY_LOCATION:

        #get the time series for this city
        rainfall_series = data.get_rain_city(city)
        #get the autocorrelation and partial autocorrelation of the time series
        acf = stattools.acf(rainfall_series, nlags=20, fft=True)
        pacf = stattools.pacf(rainfall_series, nlags=20)

        #plot the time series
        plt.figure()
        plt.plot(time, rainfall_series)
        plt.title(city+": precipitation")
        plt.xlabel("time")
        plt.ylabel("precipitation (" + data.rain_units + ")")
        plt.savefig(path.join(series_dir, "rainfall_" + city + ".pdf"))
        plt.close()

        #plot cdf
        n = len(rainfall_series)
        rain_sorted = np.sort(rainfall_series)
        cdf = np.asarray(range(n))
        plt.figure()
        plt.plot(rain_sorted, cdf)
        if np.any(rain_sorted == 0):
            non_zero_index = np.nonzero(rain_sorted)[0][0] - 1
            plt.scatter(0, cdf[non_zero_index])
        plt.title(city+": precipitation")
        plt.ylabel("precipitation (" + data.rain_units + ")")
        plt.ylabel("cumulative frequency")
        plt.savefig(path.join(series_dir, "rainfall_cdf_" + city + ".pdf"))
        plt.close()

        #plot the acf
        plt.figure()
        plt.bar(np.asarray(range(acf.size)), acf)
        plt.axhline(1/math.sqrt(len(time)), linestyle="--")
        plt.title(city+": autocorrelation of precipitation")
        plt.xlabel("lag (day)")
        plt.ylabel("autocorrelation")
        plt.savefig(path.join(series_dir, "rainfall_acf_" + city + ".pdf"))
        plt.close()

        #plot the pacf
        plt.figure()
        plt.bar(np.asarray(range(pacf.size)), pacf)
        plt.axhline(1/math.sqrt(len(time)), linestyle="--")
        plt.title(city+": partial autocorrelation of precipitation")
        plt.xlabel("lag (day)")
        plt.ylabel("partial autocorrelation")
        plt.savefig(path.join(series_dir, "rainfall_pacf_" + city + ".pdf"))
        plt.close()

        #plot correlation between pair of locations
            #choose a city, then work out correlation with every point
        cross_correlation = spatial_cross_correlation(
            rainfall_series, data.rain)
        figure_title = "Precipitation cross correlation with " + city
        figure_path = path.join(rain_dir, "cross_"+city+".pdf")
        message = HeatmapPlotMessage(longitude_grid,
                                     latitude_grid,
                                     cross_correlation,
                                     figure_title,
                                     figure_path)
        message.print()

    #plot the rain (spatial map for each time step)
    message_array = []
    for i in range(365):
        rain_plot = data.rain[i].copy()
        rain_plot.mask[rain_plot == 0] = True
        figure_title = ("precipitation (" + data.rain_units + ") : "
            + str(data.time_array[i]))
        path_to_figure = path.join(map_dir, str(i) + ".png")

        message = HeatmapPlotMessage(longitude_grid,
                                     latitude_grid,
                                     rain_plot,
                                     figure_title,
                                     path_to_figure)
        message_array.append(message)
    pool.map(HeatmapPlotMessage.print, message_array)

class HeatmapPlotMessage(object):

    def __init__(
        self, longitude_grid, latitude_grid, value, title, path_to_figure):
        self.longitude_grid = longitude_grid
        self.latitude_grid = latitude_grid
        self.value = value
        self.title = title
        self.path_to_figure = path_to_figure

    def print(self):
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(self.longitude_grid,
                       self.latitude_grid,
                       self.value)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title(self.title)
        plt.savefig(self.path_to_figure)
        plt.close()


def plot_model_fields(data, figure_dir, pool):
    """Plot figures for the model fields

    Plots:
        For each time (first year for now):
            observed model fields (heatmap) on coarse grid
            observed model fields (heatmap) on fine grid
        For each city (intepolationg required):
            model fields time series
            acf
            pacf

    Args:
        data: dataset.data object
        figure_dir: where to save the figures
    """

    mf_dir = path.join(figure_dir, "model_field")
    if not path.isdir(mf_dir):
        os.mkdir(mf_dir)
    coarse_dir = path.join(mf_dir, "coarse")
    if not path.isdir(coarse_dir):
        os.mkdir(coarse_dir)
    fine_dir = path.join(mf_dir, "fine")
    if not path.isdir(fine_dir):
        os.mkdir(fine_dir)
    series_dir = path.join(mf_dir, "series")
    if not path.isdir(series_dir):
        os.mkdir(series_dir)
    cross_dir = path.join(mf_dir, "cross")
    if not path.isdir(cross_dir):
        os.mkdir(cross_dir)

    time = data.time_array
    angle_resolution = dataset.ANGLE_RESOLUTION
    latitude_grid = data.topography["latitude"] + angle_resolution / 2
    longitude_grid = data.topography["longitude"] - angle_resolution / 2
    latitude_coarse_grid = data.topography_coarse["latitude"]
    longitude_coarse_grid = data.topography_coarse["longitude"]
    units = data.model_field_units

    fine_mean_dir = path.join(fine_dir, "mean")
    if not path.isdir(fine_mean_dir):
        os.mkdir(fine_mean_dir)
    coarse_mean_dir = path.join(coarse_dir, "mean")
    if not path.isdir(coarse_mean_dir):
        os.mkdir(coarse_mean_dir)

    #create array to loop over different grids (fine grid then coarse grid)
    latitude_array = [latitude_grid, latitude_coarse_grid]
    longitude_array = [longitude_grid, longitude_coarse_grid]
    model_field_array = [data.model_field, data.model_field_coarse]
    dir_array = [fine_mean_dir, coarse_mean_dir]

    #for fine grid, then coarse grid
    for i, dir in enumerate(dir_array):
        #for each model field
        for model_field, value in model_field_array[i].items():
            model_field_mean = np.mean(value, 0)
            #plot the mean model field (over time) as a heat map
            plt.figure()
            ax = plt.axes(projection=crs.PlateCarree())
            im = ax.pcolor(longitude_array[i],
                           latitude_array[i],
                           model_field_mean)
            ax.coastlines(resolution="50m")
            plt.colorbar(im)
            ax.set_aspect("auto", adjustable=None)
            plt.title("mean " + model_field + " (" + units[model_field] + ")")
            plt.savefig(path.join(dir, model_field + "_mean.pdf"))
            plt.close()

    fine_map_dir = path.join(fine_dir, "map")
    if not path.isdir(fine_map_dir):
        os.mkdir(fine_map_dir)
    coarse_map_dir = path.join(coarse_dir, "map")
    if not path.isdir(coarse_map_dir):
        os.mkdir(coarse_map_dir)
    dir_array  = [fine_map_dir, coarse_map_dir]

    #for each grid (eg coarse and fine)
    for i, dir in enumerate(dir_array):
        #for each model field
        for model_field, value in model_field_array[i].items():
            #plot model field for each day in parallel
            message_array = []
            for i_time in range(365):
                #heatmap plot
                figure_title = (model_field + " (" + units[model_field] + ") : "
                    + str(data.time_array[i_time]))
                path_to_figure = path.join(
                    dir, model_field + "_" + str(i_time) + ".png")
                message = HeatmapPlotMessage(longitude_array[i],
                                             latitude_array[i],
                                             value[i_time],
                                             figure_title,
                                             path_to_figure)
                message_array.append(message)
            pool.map(HeatmapPlotMessage.print, message_array)

    #for each city time series
    for city in dataset.CITY_LOCATION:

        #get the time series
        model_field_time_series = data.get_model_field_city(city)

        for model_field, time_series in model_field_time_series.items():

            time_series = np.asarray(time_series)
            units = data.model_field_units[model_field]
            cross_correlation = spatial_cross_correlation(
                time_series, data.model_field[model_field])

            #get the acf and pacf
            #in the model fields, 4 readings per day, want to have 1.5 years of
                #lag to look for seasonality
            acf = stattools.acf(time_series, nlags=10, fft=True)
            pacf = stattools.pacf(time_series, nlags=10)

            #plot the model field as a time series
            plt.figure()
            plt.plot(time, time_series)
            plt.title(city + ": " + model_field)
            plt.xlabel("time")
            plt.ylabel(model_field + " (" + units + ")")
            plt.savefig(
                path.join(series_dir, model_field + "_" + city + ".pdf"))
            plt.close()

            #plot the autocorrelation of the time series
            plt.figure()
            plt.bar(np.array(range(acf.size)), acf)
            plt.title(city + ": autocorrelation of " + model_field)
            plt.xlabel("lag (day)")
            plt.ylabel("autocorrelation")
            plt.savefig(
                path.join(series_dir, model_field + "_acf_" + city + ".pdf"))
            plt.close()

            #plot the partial autocorrelation of the time series
            plt.figure()
            plt.bar(np.array(range(pacf.size)), pacf)
            plt.title(city + ": partial autocorrelation of " + model_field)
            plt.xlabel("lag (day)")
            plt.ylabel("partial autocorrelation")
            plt.savefig(
                path.join(series_dir, model_field + "_pacf_" + city + ".pdf"))
            plt.close()

            #plot correlation between pairs of locations
            figure_title = model_field + " cross correlation with " + city
            figure_path = path.join(
                cross_dir, model_field+"_cross_"+city+".pdf")
            message = HeatmapPlotMessage(longitude_array[0],
                                         latitude_array[0],
                                         cross_correlation,
                                         figure_title,
                                         figure_path)
            message.print()

def plot_matrix(data, figure_dir):
    """Plot figures for the topography

    The model fields will need to be interpolated

    Args:
        data: dataset.data object
        figure_dir: where to save the figures
    """
    figure_dir = path.join(figure_dir, "matrix")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    #for each city, do matrix plot of all the variables
    for city in dataset.CITY_LOCATION:

        data_frame = {}
        data_frame["rain"] = np.asarray(data.get_rain_city(city))
        data_frame = pd.DataFrame(data_frame)

        model_field = data.get_model_field_city(city)
        data_frame = data_frame.join(model_field)
        #matrix plot
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        pd.plotting.scatter_matrix(data_frame, s=5, ax=ax)
        plt.savefig(path.join(figure_dir, city + ".png"))
        plt.close()

def plot_topography(data, figure_dir):
    """Matrix plot for the precipitation and model fields for each city

    Both on fine grid and coarse grid, interpolate from fine grid to coarse grid

    Args:
        data: dataset.data object
        figure_dir: where to save the figures
    """
    topo_dir = path.join(figure_dir, "topography")
    if not path.isdir(topo_dir):
        os.mkdir(topo_dir)
    coarse_dir = path.join(topo_dir, "coarse")
    if not path.isdir(coarse_dir):
        os.mkdir(coarse_dir)
    fine_dir = path.join(topo_dir, "fine")
    if not path.isdir(fine_dir):
        os.mkdir(fine_dir)

    angle_resolution = dataset.ANGLE_RESOLUTION
    latitude_grid = data.topography["latitude"] + angle_resolution / 2
    longitude_grid = data.topography["longitude"] - angle_resolution / 2
    latitude_coarse_grid = data.topography_coarse["latitude"]
    longitude_coarse_grid = data.topography_coarse["longitude"]
    latitude_array = [latitude_grid, latitude_coarse_grid]
    longitude_array = [longitude_grid, longitude_coarse_grid]

    topo_array = [data.topography, data.topography_coarse]
    dir_array = [fine_dir, coarse_dir]

    #topography
    for i, topo in enumerate(topo_array):
        for key, value in topo.items():
            plt.figure()
            ax = plt.axes(projection=crs.PlateCarree())
            im = ax.pcolor(longitude_array[i], latitude_array[i], value)
            ax.coastlines(resolution="50m")
            plt.colorbar(im)
            ax.set_aspect("auto", adjustable=None)
            plt.title(key)
            plt.savefig(path.join(dir_array[i], "topo_" + key + ".pdf"))
            plt.close()

def spatial_cross_correlation(reference_series, value_map):
    """Corrleation of one location with every other location

    Args:
        reference_series: the time series of a location to be comapred with
        value_map: dim 0: time, dim 1 and 2: 2d map, values of each location and
            time

    Return:
        2d array, same shape as value_map[0, :, :], contains correlations
    """
    cross_correlation = value_map[0].copy()
    if type(cross_correlation) is ma.core.MaskedArray:
        mask = cross_correlation.mask
    else:
        mask = np.zeros(cross_correlation.shape, dtype=np.bool_)
    for i in range(cross_correlation.shape[0]):
        for j in range(cross_correlation.shape[1]):
            if not mask[i, j]:
                cross_correlation[i, j] = scipy.stats.pearsonr(
                    reference_series, value_map[:, i, j])[0]
    return cross_correlation
