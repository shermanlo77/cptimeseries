#script exploratory analysis
#
#variables looked at: rainfall, model fields over space and time
#cities looked at: London, Cardiff, Edinburgh, Belfast, Dublin
#
#plot mean (over time) for each point in space (as a heat map)
#plot time series for each city (along with acf and pacf
#scatter plot yesterday and today rainfall for each city
#matrix plot of all the variables for each city

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools as stats
import pandas as pd
import pandas.plotting as plotting

from netCDF4 import Dataset, num2date
from pandas.plotting import register_matplotlib_converters

#required so that python datetime can be converted and can be plotted on a graph
register_matplotlib_converters()

#FUNCTION: FIND NEAREST LATITUDE AND LONGITUDE
#Given coordinates of a place, returns the nearest latitude and longitude for a
    #specific grid
#PARAMETERS:
    #coordinates: 2-element [latitude, longitude]
    #latitude_array: array of latitudes for the grid
    #longitude_array: array of longitudes for the grid
#RETURN:
    #latitude_index: pointer to latitude_array
    #longitude_index: pointer to longitude_array
def find_nearest_latitude_longitude(coordinates,
    latitude_array, longitude_array):
    #find longitude and latitude from the grid closest to coordinates
    min_longitude_error = float("inf")
    min_latitude_error = float("inf")
    #find latitude
    for i, latitude in enumerate(latitude_array):
        latitude_error = abs(latitude - coordinates[0])
        if min_latitude_error > latitude_error:
            latitude_index = i
            min_latitude_error = latitude_error
    #find longitude
    for i, longitude in enumerate(longitude_array):
        longitude_error = abs(longitude - coordinates[1])
        if min_longitude_error > longitude_error:
            longitude_index = i
            min_longitude_error = longitude_error
    return(latitude_index, longitude_index)

#latitude and longitude of cities
city_location = {
    "London": [51.5074, -0.1278],
    "Cardiff": [51.4816 + 0.1, -3.1791], #+0.1 to avoid masking a coastal city
    "Edinburgh": [55.9533, -3.1883],
    "Belfast": [54.5973, -5.9301],
    "Dublin": [53.3498, -6.2603],
}

##########          RAIN DATA          ##########

#load the .nc file using netCDF4
file_name = "Data/Rain_Data_Nov19/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"
rain_data = Dataset(file_name, "r", format="NETCDF4")
rain_data = rain_data.variables

#get the mean rainfall (mean over time) for each point in space
mean_rainfall = np.mean(rain_data["rr"][:],0)
#get the longitude and latitude
longitude_array = np.round_(rain_data["longitude"], 2)
latitude_array = np.round_(rain_data["latitude"], 2)
#for plotting, get a grid of longitudes and latitudes
longitude_grid, latitude_grid = np.meshgrid(longitude_array, latitude_array)
#get the times
time = rain_data["time"]
time = num2date(time[:], time.units)

#heatmap plot the mean rainfall
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
im = ax.pcolor(longitude_grid, latitude_grid, mean_rainfall)
ax.coastlines(resolution="50m")
plt.colorbar(im)
ax.set_aspect("auto", adjustable=None)
plt.title("Mean "+rain_data["rr"].long_name+" ("+rain_data["rr"].units+")")
plt.savefig("figures/rainfall_mean.png")
plt.close()

#plot the rainfall as a time series for each city
for city, coordinates in city_location.items():
    #get coordinates for this city (select a specific longitude and latitude)
    latitude_index, longitude_index = find_nearest_latitude_longitude(
        coordinates, latitude_array, longitude_array)
    #get the time series for this city
    rainfall_series = rain_data["rr"][:,latitude_index,longitude_index]
    #get the autocorrelation and partial autocorrelation of the time series
    acf = stats.acf(rainfall_series, nlags=100, fft=True)
    pacf = stats.pacf(rainfall_series, nlags=10)
    
    #plot the time series
    plt.figure()
    plt.plot(time, rainfall_series)
    plt.title(city+": Rain")
    plt.xlabel("Time (calendar year)")
    plt.ylabel(rain_data["rr"].long_name+" ("+rain_data["rr"].units+")")
    plt.savefig("figures/rainfall_"+city+".png")
    plt.close()
    
    #plot the acf
    plt.figure()
    plt.bar(np.asarray(range(acf.size)), acf)
    plt.title(city+": Autocorrelation of rain")
    plt.xlabel("Lag (day)")
    plt.ylabel("Autocorrelation")
    plt.savefig("figures/rainfall_acf_"+city+".png")
    plt.close()
    
    #plot the pacf
    plt.figure()
    plt.bar(np.asarray(range(pacf.size)), pacf)
    plt.title(city+": Partial autocorrelation of rain")
    plt.xlabel("Lag (day)")
    plt.ylabel("Partial autocorrelation")
    plt.savefig("figures/rainfall_pacf_"+city+".png")
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

#find the grid cooridnates of each city, save it in city_location_index
city_location_index = {} #key: city name, value: coordinates
for city, coordinates in city_location.items():
    latitude_index, longitude_index = find_nearest_latitude_longitude(
        coordinates, latitude_array, longitude_array)
    city_location_index[city] = [latitude_index, longitude_index]

#for each model field
for model_field in model_fields_data.values():
    
    #some of the model_field include time and coordinates
    #only interested in spatial temporal data, 3 dimensions
    if len(model_field.shape)==3:
        
        #get the name of the model fields
        model_field_name = model_field.long_name
        model_field_name = model_field_name.replace(".", "_")
        #untitled model fields start with "UNKNOWN"
        #add units to the name of model fields if they are named
        if model_field_name[0:7] != "UNKNOWN":
            model_field_name = model_field_name + " (" + model_field.units + ")"
        
        #get the mean (over time) model field
        model_field_array = np.asarray(model_field)
        model_field_mean = np.mean(model_field_array, 0)

        #get the resolution of the time (in days)
        time_diff = time[1] - time[0]
        time_diff = time_diff.seconds / (60 * 60 * 24)

        #plot the mean model field (over time) as a heat map
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolor(longitude_grid, latitude_grid, model_field_mean)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title("mean " + model_field_name)
        plt.savefig("figures/"+model_field_name.replace(" ", "_")+"_mean.png")
        plt.close()
        
        #for each city time series
        for city, coordinate_index in city_location_index.items():
            
            #get the coordinates of this city
            latitude_index = coordinate_index[0]
            longitude_index = coordinate_index[1]
            
            #get the timne series
            model_field_time_series = \
                model_field_array[:,latitude_index,longitude_index]
            #get the acf and pacf
            #in the model fields, 4 readings per day, want to have 1.5 years of
                #lag to look for seasonality
            acf = stats.acf(model_field_time_series, nlags=2192, fft=True)
            pacf = stats.pacf(model_field_time_series, nlags=10)
        
            #plot the model field as a time series
            plt.figure()
            plt.plot(time, model_field_time_series)
            plt.title(city+": "+model_field_name)
            plt.xlabel("Time (calendar year)")
            plt.ylabel(model_field_name)
            plt.savefig("figures/"+model_field_name.replace(" ", "_")
                +"_"+city+".png")
            plt.close()
            
            #plot the autocorrelation of the time series
            plt.figure()
            plt.bar(np.array(range(acf.size)) * time_diff, acf, width=time_diff)
            plt.title(city+": Autocorrelation of "+model_field_name)
            plt.xlabel("Lag (day)")
            plt.ylabel("Autocorrelation")
            plt.savefig("figures/"+model_field_name.replace(" ", "_")+"_acf_"
                +city+".png")
            plt.close()
            
            #plot the partial autocorrelation of the time series
            plt.figure()
            plt.bar(np.array(range(pacf.size)) * time_diff, pacf,
                width=time_diff)
            plt.title(city+": Partial autocorrelation of "+model_field_name)
            plt.xlabel("Lag (day)")
            plt.ylabel("Partial autocorrelation")
            plt.savefig("figures/"+model_field_name.replace(" ", "_")+"_pacf_"
                +city+".png")
            plt.close()

##########          MATRIX PLOT          ##########

#for each captial, do matrix plot of all the variables
for city, coordinates in city_location.items():
    
    #store data, key: variable name, value: array of values for each time
    data_frame = {}
    #in model fields, 4 readings per day, take the mean over a day
    n_time_step = int(model_fields_data["time"].size/4)
    
    #get the longitude and latitude
    #round the longitiude and latitude to 2 decimal places so that they align
        #with the model fields grid
    longitude_array = np.round_(rain_data["longitude"], 2)
    latitude_array = np.round_(rain_data["latitude"], 2)
    #get coordinates for this city (select a specific longitude and latitude)
    latitude_index, longitude_index = find_nearest_latitude_longitude(
        coordinates, latitude_array, longitude_array)
    #save the rain time series for this city
    #use only the times specified by the model fields file
    data_frame["rr"] = np.asarray(rain_data["rr"]\
        [0:n_time_step,latitude_index,longitude_index])
    
    #get the longitude and latitude in the model fields grid
    longitude_array = np.asarray(model_fields_data["longitude"])
    latitude_array = np.asarray(model_fields_data["latitude"])
    #get coordinates for this city (select a specific longitude and latitude)
    latitude_index, longitude_index = find_nearest_latitude_longitude(
        coordinates, latitude_array, longitude_array)
    
    #for each model field, put it in data_frame
    for model_field in model_fields_data.values():
        #some of the model_field include time and get_coordinates
        #interested in spatial temporal data, 3 dimensions
        if len(model_field.shape)==3:
            #get the name of the model fields
            model_field_name = model_field.long_name
            #shorten untitled model field names
            if model_field_name[0:7] == "UNKNOWN":
                model_field_name = model_field_name[20:len(model_field_name)]
                model_field_name = model_field_name.replace(".", "_")
            #get the time series
            model_field_array = model_field[:,latitude_index,longitude_index]
            #declare a vector for the shorten time series
            model_field_daily = np.zeros(n_time_step)
            #take the mean over the 4 daily readings
            for i in range(model_field_daily.size):
                model_field_daily[i] = np.mean(model_field_array[4*i:4*(i+1)])
            #save the daily mean model fields
            data_frame[model_field_name] = model_field_daily
            
            cross_correlation = stats.ccf(data_frame["rr"], model_field_daily)
            plt.figure()
            plt.plot(cross_correlation[0:730])
            plt.title(city+": Cross correlation of "+model_field_name)
            plt.ylabel("Cross correlation")
            plt.xlabel("Lag (day)")
            plt.savefig("figures/cross_correlation_"+model_field_name
                +"_"+city+".png")
            plt.close()
            
    
    #save time as a variable, integer from first time
    data_frame["time"] = np.asarray(range(n_time_step))
    
    #matrix plot
    data_frame = pd.DataFrame(data_frame)
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    plotting.scatter_matrix(data_frame, s=5, ax=ax)
    plt.savefig("figures/matrix_"+city+".png")
    plt.close()
    
    
