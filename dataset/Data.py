import math
import os

import gdal
import joblib
from netCDF4 import Dataset, num2date
import numpy as np
import numpy.ma as ma
import pandas as pd

class Data:
    
    CITY_LOCATION = {
        "London": [51.5074, -0.1278],
        #+0.1 to avoid masking a coastal city
        "Cardiff": [51.4816 + 0.1, -3.1791], 
        "Edinburgh": [55.9533, -3.1883],
        "Belfast": [54.5973, -5.9301],
        "Dublin": [53.3498, -6.2603],
    }
    LATITUDE_ARRAY = np.linspace(58.95, 49.05, 100)
    LONGITUDE_ARRAY = np.linspace(-10.95, 2.95, 140)
    RADIUS_OF_EARTH = 6371E3
    RESOLUTION = 0.1*2*math.pi*RADIUS_OF_EARTH/360
    GRAVITATIONAL_FIELD_STRENGTH = 9.81
    
    def __init__(self):
        self.model_field = None
        self.model_field_units = None
        self.rain = None
        self.mask = None
        self.rain_units = None
        self.time_array = None
        self.topography = {}
        
        longitude_grid, latitude_grid = np.meshgrid(
            Data.LONGITUDE_ARRAY, Data.LATITUDE_ARRAY)
        self.topography["longitude"] = longitude_grid
        self.topography["latitude"] = latitude_grid
    
    def copy_from(self, other):
        self.model_field = other.model_field
        self.model_field_units = other.model_field_units
        self.rain = other.rain
        self.mask = other.mask
        self.rain_units = other.rain_units
        self.time_array = other.time_array
        self.topography = other.topography
    
    def load_model_field(self, file_name):
        
        #get model fields data via netcdf4
        dataset = Dataset(file_name, "r", format="NETCDF4")
        dataset = dataset.variables
        
        #store data
            #key: variable name
            #value: 3d matrix, dim 1 and 2 for each coordinate, dim 3 for each
                #time
        self.model_field = {}
        #store units of the model fields
            #key: variable name
            #value: string
        self.model_field_units = {}
        
        self.time_array = []
        time_array = dataset["time"]
        time_array = num2date(np.asarray(time_array), time_array.units)
        
        #get the longitude and latitude in the model fields grid
        longitude_array = np.round_(np.asarray(dataset["longitude"]), 2)
        latitude_array = np.round_(np.asarray(dataset["latitude"]), 2)
        longitude_grid, latitude_grid = np.meshgrid(
            longitude_array, latitude_array)
        
        
        dataset_key = []
        
        for key, model_field in dataset.items():
            #some of the model_field include time and get_coordinates
            #interested in spatial temporal data, 3 dimensions
            if len(model_field.shape)==3:
                #check the dimension names
                if model_field.dimensions[1] != "latitude":
                    print(key, "does not have latitude as dimension 1")
                    raise
                if model_field.dimensions[2] != "longitude":
                    print(key, "does not have longitude as dimension 2")
                    raise
                #get the name of the model fields
                model_field_name = model_field.name
                dataset_key.append(key)
                #name untitled model fields
                if model_field_name[len(model_field_name)-5] == "7":
                    model_field_name = "total_column_water"
                    units = "kg m-2"
                elif model_field_name[len(model_field_name)-5] == "3":
                    model_field_name = "specific_humidity"
                    units = ""
                else:
                    units = model_field.units
                self.model_field[model_field_name] = []
                self.model_field_units[model_field_name] = units
        
        derived_model_field = [
            "wind_speed",
            "specific_humidity_rate",
            "total_column_water_rate",
        ]
        self.model_field["wind_speed"] = []
        self.model_field_units["wind_speed"] = "m s-1"
        self.model_field["specific_humidity_rate"] = []
        self.model_field_units["specific_humidity_rate"] = "s-1"
        self.model_field["total_column_water_rate"] = []
        self.model_field_units["total_column_water_rate"] = "kg m-2 s-1"
        
        n_read_per_day = 4
        
        for i in range(0, len(time_array), n_read_per_day):
            model_field_day = {}
            for i_key, model_field_name in enumerate(self.model_field.keys()):
                if not model_field_name in derived_model_field:
                    model_field_day[model_field_name] = np.asarray(
                        dataset[dataset_key[i_key]][i:i+4])
            
            model_field_day["wind_speed"] = np.sqrt(
                np.square(model_field_day["x_wind"])
                + np.square(model_field_day["y_wind"]))
            model_field_day["specific_humidity_rate"] =  self.get_rate(
                model_field_day, "specific_humidity")
            model_field_day["total_column_water_rate"] = self.get_rate(
                model_field_day, "total_column_water")
            
            for model_field_name in self.model_field.keys():
                self.model_field[model_field_name].append(
                    np.mean(model_field_day[model_field_name], 0))
            
            date_array = time_array[i:i+4]
            for i in range(n_read_per_day-1):
                if date_array[i].date() != date_array[i+1].date():
                    print("Date varies in", date_array)
                    raise
            self.time_array.append(date_array[0].date())
        
        target_longitude = Data.LONGITUDE_ARRAY
        target_latitude = Data.LATITUDE_ARRAY
        
        longitude_min = np.argwhere(
            np.isclose(longitude_array, np.min(target_longitude)))[0][0]
        longitude_max = np.argwhere(
            np.isclose(longitude_array, np.max(target_longitude)))[0][0]+1
        
        latitude_min = np.argwhere(
            np.isclose(latitude_array, np.max(target_latitude)))[0][0]
        latitude_max = np.argwhere(
            np.isclose(latitude_array, np.min(target_latitude)))[0][0]+1
        
        target_shape = self.topography["latitude"].shape
        for key, value in self.model_field.items():
            self.model_field[key] = np.asarray(value)[:,
                latitude_min:latitude_max, longitude_min:longitude_max]
            if self.model_field[key][0].shape != target_shape:
                print(key, "does not have shape", target_shape)
                raise
    
    def get_rate(self, model_field, key):
        grad = np.gradient(model_field[key], Data.RESOLUTION, axis=(1,2))
        return np.sqrt(
            np.square(grad[0] * model_field["y_wind"])
            + np.square(- grad[1] * model_field["x_wind"]))
    
    def load_rain(self, file_name):
        rain_data = Dataset(file_name, "r", format="NETCDF4")
        rain = rain_data.variables["rr"]
        self.rain_units = rain.units
        
        if rain.dimensions[1] != "latitude":
            print("Rain does not have latitude as dimension 1")
            raise
        if rain.dimensions[2] != "longitude":
            print("Rain does not have longitude as dimension 2")
            raise
        
        #get the longitude and latitude
        latitude_array = np.round_(rain_data["latitude"], 2)
        longitude_array = np.round_(rain_data["longitude"], 2)
        latitude_array = np.flip(latitude_array)
        
        rain = ma.asarray(rain[:])
        rain = np.flip(rain, 1)
        
        #get the times
        time_array = rain_data["time"]
        time_array = num2date(time_array[:], time_array.units)
        
        self.rain = []
        self.mask = rain[0].mask
        for i, time in enumerate(time_array):
            time = time.date()
            if time in self.time_array:
                self.rain.append(rain[i])
                if not np.array_equal(self.mask, rain[i].mask):
                    print("Mask in ", time, "is not consistent")
                    raise
        self.rain = ma.asarray(self.rain)
    
    def load_topo(self, file_name):
        gdal_dataset = gdal.Open(file_name)
        raster_band = gdal_dataset.GetRasterBand(1)
        topo = raster_band.ReadAsArray() / Data.GRAVITATIONAL_FIELD_STRENGTH
        topo = np.flip(topo)
        grad = np.gradient(topo, Data.RESOLUTION)
        grad = np.sqrt(np.square(grad[0]) + np.square(grad[1]))
        topo = topo[2:102, 2:142]
        grad = grad[2:102, 2:142]
        self.topography["elevation"] = np.flip(topo)
        self.topography["gradient"] = np.flip(grad)
    
    #FUNCTION: FIND NEAREST LATITUDE AND LONGITUDE
    #Given coordinates of a place, returns the nearest latitude and longitude
        #for a specific grid
    #PARAMETERS:
        #coordinates: 2-element [latitude, longitude]
        #latitude_array: array of latitudes for the grid
        #longitude_array: array of longitudes for the grid
    #RETURN:
        #latitude_index: pointer to latitude_array
        #longitude_index: pointer to longitude_array
    def find_nearest_latitude_longitude(self, coordinates):
        #find longitude and latitude from the grid closest to coordinates
        min_longitude_error = float("inf")
        min_latitude_error = float("inf")
        #find latitude
        for i, latitude in enumerate(Data.LATITUDE_ARRAY):
            latitude_error = abs(latitude - coordinates[0])
            if min_latitude_error > latitude_error:
                latitude_index = i
                min_latitude_error = latitude_error
        #find longitude
        for i, longitude in enumerate(Data.LONGITUDE_ARRAY):
            longitude_error = abs(longitude - coordinates[1])
            if min_longitude_error > longitude_error:
                longitude_index = i
                min_longitude_error = longitude_error
        return(latitude_index, longitude_index)
    
    def get_latitude_longitude_city(self, city):
        return self.find_nearest_latitude_longitude(Data.CITY_LOCATION[city])
    
    def get_latitude_longitude_random(self, rng):
        latitude_index = rng.randint(0, len(Data.LATITUDE_ARRAY))
        longitude_index = rng.randint(0, len(Data.LONGITUDE_ARRAY))
        return(latitude_index, longitude_index)
    
    def get_latitude_longitude_random_mask(self, rng):
        mask_index = np.where(np.logical_not(mask))
        random_index = rng.randint(0, len(mask_index[0]))
        latitude_index = mask_index[0][random_index]
        longitude_index = mask_index[1][random_index]
        return(latitude_index, longitude_index)
    
    def get_data(self, latitude_index, longitude_index):
        model_field = self.get_model_field(latitude_index, longitude_index)
        rain = self.get_rain(latitude_index, longitude_index)
        return (model_field, rain)
    
    def get_model_field(self, latitude_index, longitude_index):
        data_frame = {}
        for model_field_name, value in self.model_field.items():
            data_frame[model_field_name] = value[
                :, latitude_index, longitude_index]
        return pd.DataFrame(data_frame)
    
    def get_rain(self, latitude_index, longitude_index):
        return self.rain[:, latitude_index, longitude_index]
    
    def get_data_city(self, city):
        latitude_index, longitude_index = self.get_latitude_longitude_city(city)
        return self.get_data(latitude_index, longitude_index)
    
    def get_model_field_city(self, city):
        latitude_index, longitude_index = self.get_latitude_longitude_city(city)
        return self.get_model_field(latitude_index, longitude_index)
    
    def get_rain_city(self, city):
        latitude_index, longitude_index = self.get_latitude_longitude_city(city)
        return self.get_rain(latitude_index, longitude_index)
    
    def get_data_random(self, rng):
        latitude_index, longitude_index = (
            self.get_latitude_longitude_random_mask(rng))
        return self.get_data(latitude_index, longitude_index)
    
    def get_model_field_random(self, rng):
        latitude_index, longitude_index = self.get_latitude_longitude_random(
            rng)
        return self.get_model_field(latitude_index, longitude_index)
    
    def get_rain_random(self, rng):
        latitude_index, longitude_index = (
            self.get_latitude_longitude_random_mask(rng))
        return self.get_rain(latitude_index, longitude_index)
