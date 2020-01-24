import numpy as np
import pandas as pd
import math
import compound_poisson as cp
from netCDF4 import Dataset, num2date

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

def get_london_rain():
    #latitude and longitude of cities
    coordinates = [51.5074, -0.1278]
    #load the .nc file using netCDF4
    file_name = "../Data/Rain_Data_Nov19/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"
    rain_data = Dataset(file_name, "r", format="NETCDF4")
    rain_data = rain_data.variables
    #get the longitude and latitude
    longitude_array = np.round_(rain_data["longitude"], 2)
    latitude_array = np.round_(rain_data["latitude"], 2)
    #for plotting, get a grid of longitudes and latitudes
    longitude_grid, latitude_grid = np.meshgrid(longitude_array, latitude_array)

    #get coordinates for this city (select a specific longitude and latitude)
    latitude_index, longitude_index = find_nearest_latitude_longitude(
        coordinates, latitude_array, longitude_array)
    #get the time series for this city
    rainfall = rain_data["rr"][:,latitude_index,longitude_index]
    return np.asarray(rainfall)

def get_london_rain_training():
    rainfall = get_london_rain()
    return rainfall[0:3000]

def get_london_rain_test():
    rainfall = get_london_rain()
    return rainfall[3000:4018]

def get_time():
    file_name = "../Data/Rain_Data_Nov19/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"
    model_fields_data = Dataset(file_name, "r", format="NETCDF4")
    model_fields_data = model_fields_data.variables
    time = model_fields_data["time"]
    time = num2date(np.asarray(time), time.units)
    return time

def get_time_training():
    time = get_time()
    return time[0:3000]

def get_time_test():
    time = get_time()
    return time[3000:4018]

def get_london_model_field():
    #get model fields data via netcdf4
    coordinates = [51.5074, -0.1278]
    file_name = "../Data/Rain_Data_Nov19/ana_input_1.nc"
    model_fields_data = Dataset(file_name, "r", format="NETCDF4")
    model_fields_data = model_fields_data.variables

    #store data, key: variable name, value: array of values for each time
    data_frame = {}
    #in model fields, 4 readings per day, take the mean over a day
    n = model_fields_data["time"].size

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
            if model_field_name[len(model_field_name)-5] == "7":
                model_field_name = "total_column_water"
            elif model_field_name[len(model_field_name)-5] == "3":
                model_field_name = "specific humidity"
            #get the time series
            model_field_series = model_field[:,latitude_index,longitude_index]
            
            for i_quarter in range(4):
                data_frame[model_field_name+"_"+str(i_quarter)] = (
                    model_field_series[range(i_quarter, n, 4)])
    x = pd.DataFrame(data_frame)
    return x

def get_london_model_field_training():
    model_field = get_london_model_field()
    return model_field[0:3000]

def get_london_model_field_test():
    model_field = get_london_model_field()
    return model_field[3000:4018]
