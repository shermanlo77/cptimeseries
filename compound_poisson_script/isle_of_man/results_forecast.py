import os
import sys

import cartopy.crs as ccrs
import joblib
import matplotlib.pyplot as plot
import numpy as np
from numpy import ma
import pdb

sys.path.append(os.path.join("..", ".."))
import compound_poisson as cp
import dataset


def main():
    
    if not os.path.isdir(os.path.join("figure", "series_forecast")):
        os.mkdir(os.path.join("figure", "series_forecast"))
    if not os.path.isdir(os.path.join("figure", "forecast")):
        os.mkdir(os.path.join("figure", "forecast"))
    
    forecast_array = joblib.load("forecast.gz")
    test_set = dataset.IsleOfManTest()
    directory = os.path.join("figure", "series_forecast")
    forecast_map = ma.empty_like(test_set.rain)
    
    for lat_i in range(forecast_map.shape[1]):
        for long_i in range(forecast_map.shape[2]):
            if forecast_array[lat_i][long_i]:
                rain = test_set.get_rain(lat_i, long_i)
                forecast = forecast_array[lat_i][long_i]
                forecast.time_array = test_set.time_array
                cp.print.forecast(forecast,
                                  rain,
                                  directory,
                                  prefix=str(lat_i)+"_"+str(long_i))
                forecast_map[:, lat_i, long_i] = forecast.forecast_median

    #plot the rain
    longitude_grid = test_set.topography["longitude"]
    latitude_grid = test_set.topography["latitude"]
    angle_resolution = dataset.data.ANGLE_RESOLUTION
    for i, time in enumerate(test_set.time_array):
        
        rain_i = forecast_map[i, :, :]
        rain_i.mask[rain_i == 0] = True
        
        plot.figure()
        ax = plot.axes(projection=ccrs.PlateCarree())
        im = ax.pcolor(longitude_grid - angle_resolution / 2,
                       latitude_grid + angle_resolution / 2,
                       rain_i,
                       vmin=0,
                       vmax=50)
        ax.coastlines(resolution="50m")
        plot.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plot.title("precipitation (" + test_set.rain_units + ") : " + str(time))
        plot.savefig(os.path.join("figure", "forecast", str(i) + ".png"))
        plot.close()

if __name__ == "__main__":
    main()
