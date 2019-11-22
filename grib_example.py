#example of reading a .grib file
#
#uses the data ana_coarse.grib, contains a snapshot of multiple model fields
#plots each model field as a heatmap

import cartopy.crs as ccrs
import gdal
import gmplot
import matplotlib.pyplot as plt
import numpy as np
import pupygrib

#location of the ana.grib file
file_name = 'Data/Rain_Data/ana_coarse.grib'
file_stream = open(file_name, 'rb')

#pupygrib for reading the data
message_iterator = pupygrib.read(file_stream)
#gdal for reading the meta data
gdal_dataset = gdal.Open(file_name)

#for each variable
for i, message in enumerate(message_iterator):
    
    #get longtitude and latitude
    #longtitude and latitude are meshgrid-like matrices
    longitude, latitude = message.get_coordinates()
    #get values, matrix format
    values = message.get_values()
    
    #get description of the data from the meta data
    raster_band = gdal_dataset.GetRasterBand(i+1)
    variable_name = raster_band.GetMetadata()['GRIB_COMMENT']
    
    #plot the heatmap on a coastline map
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolor(longitude, latitude, values)
    ax.coastlines()
    plt.colorbar(im)
    plt.title(variable_name)
    ax.set_aspect('auto', adjustable=None)
    plt.show()
    plt.close()

file_stream.close()
