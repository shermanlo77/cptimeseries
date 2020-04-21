#!/usr/bin/python

#script exploratory analysis
#
#variables looked at: rainfall, model fields over space and time
#cities looked at: London, Cardiff, Edinburgh, Belfast, Dublin
#
#plot mean (over time) for each point in space (as a heat map)
#plot time series for each city (along with acf and pacf
#scatter plot yesterday and today rainfall for each city
#matrix plot of all the variables for each city

import math
import os
from os import path
import pathlib

from cartopy import crs
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import pandas as pd
from statsmodels.tsa import stattools

import dataset

def plot_data(data):
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

    plot_rain(data, figure_dir)
    plot_model_fields(data, figure_dir)
    plot_matrix(data, figure_dir)
    plot_topography(data, figure_dir)

def plot_rain(data, figure_dir):
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

    #plot the rain
    for i in range(365):

        rain_plot = data.rain[i].copy()
        rain_plot.mask[rain_plot == 0] = True

        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid,
                       latitude_grid,
                       rain_plot,
                       vmin=0,
                       vmax=50)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title(
            "precipitation (" + data.rain_units + ") : "
            + str(data.time_array[i]))
        plt.savefig(path.join(map_dir, str(i) + ".png"))
        plt.close()

def plot_model_fields(data, figure_dir):
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

    latitude_array = [latitude_grid, latitude_coarse_grid]
    longitude_array = [longitude_grid, longitude_coarse_grid]
    model_field_array = [data.model_field, data.model_field_coarse]
    dir_array = [fine_mean_dir, coarse_mean_dir]

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

    for i, dir in enumerate(dir_array):
        #for each model field
        for model_field, value in model_field_array[i].items():
            for i_time in range(365):
                #plot the mean model field (over time) as a heat map
                plt.figure()
                ax = plt.axes(projection=crs.PlateCarree())
                im = ax.pcolor(longitude_array[i],
                               latitude_array[i],
                               value[i_time])
                ax.coastlines(resolution="50m")
                plt.colorbar(im)
                ax.set_aspect("auto", adjustable=None)
                plt.title(
                    model_field + " (" + units[model_field] + ") : "
                    + str(data.time_array[i_time]))
                plt.savefig(
                    path.join(dir, model_field + "_" + str(i_time) + ".png"))
                plt.close()

    #for each city time series
    for city in dataset.CITY_LOCATION:

        #get the time series
        model_field_time_series = data.get_model_field_city(city)

        for model_field, time_series in model_field_time_series.items():

            time_series = np.asarray(time_series)
            units = data.model_field_units[model_field]

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

def plot_matrix(data, figure_dir):
    ##########          MATRIX PLOT          ##########
    figure_dir = path.join(figure_dir, "matrix")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    #for each captial, do matrix plot of all the variables
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
