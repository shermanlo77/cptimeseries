#!/usr/bin/python

import os
from os import path
import pathlib

from cartopy import crs
import matplotlib.pyplot as plt
import numpy as np

import dataset

def main():

    path_here = pathlib.Path(__file__).parent.absolute()
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    figure_dir = path.join(figure_dir, "coarse")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)

    data = dataset.AnaDualExample1()
    time = data.time_array
    latitude_grid = data.topography_coarse["latitude"]
    longitude_grid = data.topography_coarse["longitude"]

    #for each model field
    for model_field, value in data.model_field_coarse.items():

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

        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        if model_field == "air_temperature":
            im = ax.pcolor(
                longitude_grid, latitude_grid, value[0], vmin=256, vmax=265)
        else:
            im = ax.pcolor(longitude_grid, latitude_grid, value[0])
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title(model_field + " (" + units + ")")
        plt.savefig(path.join(figure_dir, model_field + "_0.pdf"))
        plt.close()

    for key, value in data.topography_coarse.items():
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid, latitude_grid, value)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title(key)
        plt.savefig(path.join(figure_dir, "topo_" + key + ".pdf"))
        plt.close()

    for key, value in data.topography_coarse_normalise.items():
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid, latitude_grid, value)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title(key)
        plt.savefig(path.join(figure_dir, "topo_normalise_" + key + ".pdf"))
        plt.close()

if __name__ == "__main__":
    main()
