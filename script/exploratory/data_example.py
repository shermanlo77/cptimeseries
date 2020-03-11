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
import pandas as pd
from statsmodels.tsa import stattools

import dataset

def main():
    #required so that python datetime can be converted and can be plotted on a
        #graph
    pd.plotting.register_matplotlib_converters()
    
    path_here = pathlib.Path(__file__).parent.absolute()
    figure_dir = path.join(path_here, "figure")
    rain_dir = path.join(figure_dir, "rain")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    if not path.isdir(rain_dir):
        os.mkdir(rain_dir)

    data = dataset.Ana_1()
    time = data.time_array
    latitude_grid = data.topography["latitude"]
    longitude_grid = data.topography["longitude"]
    angle_resolution = dataset.ANGLE_RESOLUTION

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
    plt.savefig(path.join(figure_dir, "rainfall_mean.pdf"))
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
        plt.savefig(path.join(figure_dir, "rainfall_" + city + ".pdf"))
        plt.close()
        
        #plot the acf
        plt.figure()
        plt.bar(np.asarray(range(acf.size)), acf)
        plt.axhline(1/math.sqrt(len(time)), linestyle="--")
        plt.title(city+": autocorrelation of rain")
        plt.xlabel("lag (day)")
        plt.ylabel("autocorrelation")
        plt.savefig(path.join(figure_dir, "rainfall_acf_" + city + ".pdf"))
        plt.close()
        
        #plot the pacf
        plt.figure()
        plt.bar(np.asarray(range(pacf.size)), pacf)
        plt.axhline(1/math.sqrt(len(time)), linestyle="--")
        plt.title(city+": partial autocorrelation of rain")
        plt.xlabel("lag (day)")
        plt.ylabel("partial autocorrelation")
        plt.savefig(path.join(figure_dir, "rainfall_pacf_" + city + ".pdf"))
        plt.close()
    
    #plot the rain
    for i in range(365):
        
        rain_plot = data.rain[i].copy()
        rain_plot.mask[rain_plot == 0] = True
        
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid - angle_resolution / 2,
                       latitude_grid + angle_resolution / 2,
                       rain_plot,
                       vmin=0,
                       vmax=50)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title(
            "precipitation (" + data.rain_units + ") : "
            + str(data.time_array[i]))
        plt.savefig(path.join(rain_dir, str(i) + ".png"))
        plt.savefig(path.join(rain_dir, str(i) + ".pdf"))
        plt.close()
    
    #for each model field
    for model_field, value in data.model_field.items():
        
        model_field_mean = np.mean(value, 0)
        units = data.model_field_units[model_field]
        
        #plot the mean model field (over time) as a heat map
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid, latitude_grid, model_field_mean)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title("mean " + model_field + " (" + units + ")")
        plt.savefig(path.join(figure_dir, model_field + "_mean.pdf"))
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
                path.join(figure_dir, model_field + "_" + city + ".pdf"))
            plt.close()
            
            #plot the autocorrelation of the time series
            plt.figure()
            plt.bar(np.array(range(acf.size)), acf)
            plt.title(city + ": autocorrelation of " + model_field)
            plt.xlabel("lag (day)")
            plt.ylabel("autocorrelation")
            plt.savefig(
                path.join(figure_dir, model_field + "_acf_" + city + ".pdf"))
            plt.close()
            
            #plot the partial autocorrelation of the time series
            plt.figure()
            plt.bar(np.array(range(pacf.size)), pacf)
            plt.title(city + ": partial autocorrelation of " + model_field)
            plt.xlabel("lag (day)")
            plt.ylabel("partial autocorrelation")
            plt.savefig(
                path.join(figure_dir, model_field + "_pacf_" + city + ".pdf"))
            plt.close()

    ##########          MATRIX PLOT          ##########

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
        plt.savefig(path.join(figure_dir, "matrix_" + city + ".png"))
        plt.close()
    
    #topography
    for key, value in data.topography.items():
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid, latitude_grid, value)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title(key)
        plt.savefig(path.join(figure_dir, "topo_" + key + ".pdf"))
        plt.close()
    
if __name__ == "__main__":
    main()
