import os
from os import path
import pathlib

from cartopy import crs
import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

import compound_poisson
import dataset

def main():
    
    path_here = pathlib.Path(__file__).parent.absolute()
    
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    series_dir = path.join(figure_dir, "series_forecast")
    if not path.isdir(series_dir):
        os.mkdir(series_dir)
    forecast_dir = path.join(figure_dir, "forecast")
    if not path.isdir(forecast_dir):
        os.mkdir(forecast_dir)
    
    forecast_array = joblib.load(path.join(path_here, "forecast.gz"))
    test_set = dataset.IsleOfManTest()
    forecast_map = ma.empty_like(test_set.rain)
    
    for lat_i in range(forecast_map.shape[1]):
        for long_i in range(forecast_map.shape[2]):
            if forecast_array[lat_i][long_i]:
                rain = test_set.get_rain(lat_i, long_i)
                forecast = forecast_array[lat_i][long_i]
                forecast.time_array = test_set.time_array
                compound_poisson.print.forecast(
                    forecast, rain, series_dir,
                    prefix=str(lat_i)+"_"+str(long_i))
                forecast_map[:, lat_i, long_i] = forecast.forecast_median

    #plot the rain
    longitude_grid = test_set.topography["longitude"]
    latitude_grid = test_set.topography["latitude"]
    angle_resolution = dataset.ANGLE_RESOLUTION
    for i, time in enumerate(test_set.time_array):
        
        rain_i = forecast_map[i, :, :]
        rain_i.mask[rain_i == 0] = True
        
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid - angle_resolution / 2,
                       latitude_grid + angle_resolution / 2,
                       rain_i,
                       vmin=0,
                       vmax=50)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title("precipitation (" + test_set.rain_units + ") : " + str(time))
        plt.savefig(path.join(forecast_dir, str(i) + ".png"))
        plt.close()

if __name__ == "__main__":
    main()
