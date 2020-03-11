import os
from os import path
import pathlib

from cartopy import crs
import matplotlib.pyplot as plt

import dataset

def main():
    
    path_here = pathlib.Path(__file__).parent.absolute()
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    figure_dir = path.join(figure_dir, "isle_of_man")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    
    data = dataset.IsleOfMan()
    latitude_grid = data.topography["latitude"]
    longitude_grid = data.topography["longitude"]
    angle_resolution = dataset.ANGLE_RESOLUTION
    
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
        plt.savefig(path.join(figure_dir, str(i) + ".png"))
        plt.close()

if __name__ == "__main__":
    main()
