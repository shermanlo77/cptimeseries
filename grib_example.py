import cartopy.crs as ccrs
import gdal
import gmplot
import matplotlib.pyplot as plt
import numpy as np
import pupygrib

#dataset = gdal.Open('Data/Rain_Data/ana_coarse.grib')
#number_bands = dataset.RasterCount
#channel = dataset.GetRasterBand(1).ReadAsArray()

#print(dataset)
#print(number_bands)
#print(channel)

#open the grib file using pupygrib
file_stream = open('Data/Rain_Data/ana_coarse.grib','rb')
grib_iterator = pupygrib.read(file_stream)
#for each variable
for i, message in enumerate(grib_iterator):

    #get longtitude and latitude
    #longtitude and latitude are meshgrid-like matrices
    longitude, latitude = message.get_coordinates()
    #get values, matrix format
    values = message.get_values()

    #plot the heatmap on a map
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolor(longitude, latitude, values)
    ax.coastlines()
    plt.colorbar(im)
    ax.set_aspect('auto', adjustable=None)
    plt.show()

file_stream.close()