import os
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

sys.path.append("..")
import dataset

def main():
    
    if not os.path.isdir("figure"):
        os.mkdir("figure")
    if not os.path.isdir(os.path.join("figure", "isle_of_man")):
        os.mkdir(os.path.join("figure", "isle_of_man"))
    
    data = dataset.IsleOfMan()
    latitude_grid = data.topography["latitude"]
    longitude_grid = data.topography["longitude"]
    angle_resolution = dataset.Data.ANGLE_RESOLUTION
    
    #plot the rain
    for i in range(365):
        
        rain_plot = data.rain[i].copy()
        rain_plot.mask[rain_plot == 0] = True
        
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
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
        plt.savefig(
            os.path.join("figure", "isle_of_man", str(i) + ".png"))
        plt.close()

if __name__ == "__main__":
    main()
