#script for reading in the .nc file and ana_input_1.grib file
#
#plot the mean (over time) rainfall
#plot the rainfall time series in London
#
#for each model fields:
    #plot the mean (over time)
    #plot the time series in London

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from netCDF4 import Dataset, num2date
from pandas.plotting import register_matplotlib_converters

#required so that python datetime can be converted and can be plotted on a graph
register_matplotlib_converters()

##########          RAIN DATA          ##########

#load the .nc file using netCDF4
file_name = "Data/Rain_Data_Nov19/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"
rain_data = Dataset(file_name, "r", format="NETCDF4")
rain_data = rain_data.variables

#get the mean rainfall (mean over time) for each point in space
mean_rainfall = np.mean(rain_data['rr'][:],0)
#get the longitude and latitude
longitude_array = rain_data['longitude']
latitude_array = rain_data['latitude']
#for plotting, get a grid of longitudes and latitudes
longitude_grid, latitude_grid = np.meshgrid(longitude_array, latitude_array)
#get the times
time = rain_data['time']
time = num2date(time[:], time.units)

#get cooridnates for London (select a specific longitude and latitude)
#London: 51.5074 deg N, 0.1278 deg W
min_longitude_error = float("inf")
for i, longitude in enumerate(longitude_array):
    longitude_error = abs(longitude - 0.1278)
    if min_longitude_error > longitude_error:
        longitude_index = i
        min_longitude_error = longitude_error
min_latitude_error = float("inf")
for i, latitude in enumerate(latitude_array):
    latitude_error = abs(latitude - 51.5074)
    if min_latitude_error > latitude_error:
        latitude_index = i
        min_latitude_error = latitude_error

#heatmap plot the mean rainfall
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
im = ax.pcolor(longitude_grid, latitude_grid, mean_rainfall)
ax.coastlines(resolution='50m')
plt.colorbar(im)
ax.set_aspect('auto', adjustable=None)
plt.title("Mean "+rain_data['rr'].long_name+" ("+rain_data['rr'].units+")")
plt.savefig("figures/rainfall_mean.png")
plt.close()

#plot the rainfall in London
rainfall_series = rain_data['rr'][:,latitude_index,longitude_index]
plt.figure()
plt.plot(time, rainfall_series)
plt.title("London")
plt.xlabel("Time (calendar year)")
plt.ylabel(rain_data['rr'].long_name+" ("+rain_data['rr'].units+")")
plt.savefig("figures/rainfall_london.png")
plt.close()

##########          MODEL FIELDS          ##########

#get model fields data via netcdf4
file_name = "Data/Rain_Data_Nov19/ana_input_1.nc"
model_fields_data = Dataset(file_name, "r", format="NETCDF4")
model_fields_data = model_fields_data.variables

#get the longitude and latitude
longitude_array = np.asarray(model_fields_data["longitude"])
latitude_array = np.asarray(model_fields_data["latitude"])
#for plotting, get a grid of longitudes and latitudes
longitude_grid, latitude_grid = np.meshgrid(longitude_array, latitude_array)
#get the times
time = model_fields_data["time"]
time = num2date(np.asarray(time), time.units)

#get cooridnates for London (select a specific longitude and latitude)
#London: 51.5074 deg N, 0.1278 deg W
min_longitude_error = float("inf")
for i, longitude in enumerate(longitude_array):
    longitude_error = abs(longitude - 0.1278)
    if min_longitude_error > longitude_error:
        longitude_index = i
        min_longitude_error = longitude_error
min_latitude_error = float("inf")
for i, latitude in enumerate(latitude_array):
    latitude_error = abs(latitude - 51.5074)
    if min_latitude_error > latitude_error:
        latitude_index = i
        min_latitude_error = latitude_error

#plot the mean (over time) model fields
for key, model_field in model_fields_data.items():
    
    #some of the model_field include time and get_coordinates
    #interested in spatial temporal data, 3 dimensions
    if len(model_field.shape)==3:
        
        #get the name of the model fields
        model_field_name = model_field.long_name
        #untitled model fields start with "UNKNOWN"
        #add units to the name of model fields if they are named
        if model_field_name[0:7] != "UNKNOWN":
            model_field_name = model_field_name + " (" + model_field.units + ")"
        
        #get the mean (over time) model field
        model_field_array = np.asarray(model_field)
        model_field_mean = np.mean(model_field_array, 0)
        #get the model field in London
        model_field_time_series = \
            model_field_array[:,latitude_index,longitude_index]
        
        #plot the mean model field (over time) as a heat mao
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolor(longitude_grid, latitude_grid, model_field_mean)
        ax.coastlines(resolution='50m')
        plt.colorbar(im)
        ax.set_aspect('auto', adjustable=None)
        plt.title("mean " + model_field_name)
        plt.savefig("figures/"+model_field_name+"_mean.png")
        plt.close()
        
        #plot the model field in London as a time series
        plt.figure()
        plt.plot(time, model_field_time_series)
        plt.title("London")
        plt.xlabel("Time (calendar year)")
        plt.ylabel(model_field_name)
        plt.savefig("figures/"+model_field_name+"_london.png")
        plt.close()
