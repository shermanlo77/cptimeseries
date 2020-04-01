import datetime
import math
import os
from os import path
import pathlib

import gdal
import joblib
import netCDF4
import numpy as np
from numpy import ma
import pandas as pd
from scipy import interpolate

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
LATITUDE_TOPO_ARRAY = np.linspace(59.15, 48.85, 104)
LONGITUDE_TOPO_ARRAY = np.linspace(-11.15, 3.15, 144)
LATITUDE_COARSE_ARRAY = np.linspace(59.445, 48.888, 20)
LONGITUDE_COARSE_ARRAY = np.linspace(-12.5, 4.167, 21)
RADIUS_OF_EARTH = 6371E3
ANGLE_RESOLUTION = 0.1
RESOLUTION = ANGLE_RESOLUTION*2*math.pi*RADIUS_OF_EARTH/360
GRAVITATIONAL_FIELD_STRENGTH = 9.81

class Data(object):

    def __init__(self, path_to_storage=None):
        self.model_field = {}
        self.model_field_units = {}
        self.rain = []
        self.mask = None
        self.rain_units = None
        self.time_array = []
        self.topography = {}
        self.topography_normalise = {}

        if path_to_storage is None:
            path_to_storage = pathlib.Path(__file__).parent.absolute()
        storage_file =  path.join(
            path_to_storage, self.__class__.__name__+".gz")
        if path.isfile(storage_file):
            print("Loading", storage_file)
            self.copy_from(joblib.load(storage_file))
        else:
            longitude_grid, latitude_grid = np.meshgrid(
                LONGITUDE_ARRAY, LATITUDE_ARRAY)
            self.topography["longitude"] = longitude_grid
            self.topography["latitude"] = latitude_grid
            self.load_data()
            print("Saving", storage_file)
            joblib.dump(self, storage_file)

    def copy_from(self, other):
        self.model_field = other.model_field
        self.model_field_units = other.model_field_units
        self.rain = other.rain
        self.mask = other.mask
        self.rain_units = other.rain_units
        self.time_array = other.time_array
        self.topography = other.topography
        self.topography_normalise = other.topography_normalise

    def load_data(self):
        raise NotImplementedError()

    def load_model_field(self, file_name):

        #get model fields data via netcdf4
        dataset_file = netCDF4.Dataset(file_name, "r", format="NETCDF4")
        dataset = dataset_file.variables
        time_array = dataset["time"]
        time_array = netCDF4.num2date(np.asarray(time_array), time_array.units)

        #get the longitude and latitude in the model fields grid
        longitude_array = np.round_(np.asarray(dataset["longitude"]), 2)
        latitude_array = np.round_(np.asarray(dataset["latitude"]), 2)
        longitude_grid, latitude_grid = np.meshgrid(
            longitude_array, latitude_array)

        dataset_key = []

        for key, model_field in dataset.items():
            #some of the model_field include time and get_coordinates
            #interested in spatial temporal data, 3 dimensions
            if len(model_field.shape) == 3:
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
            for i_key, model_field_name in enumerate(self.model_field):
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

            for model_field_name in self.model_field:
                self.model_field[model_field_name].append(
                    np.mean(model_field_day[model_field_name], 0))

            date_array = time_array[i:i+4]
            for i in range(n_read_per_day-1):
                if date_array[i].date() != date_array[i+1].date():
                    print("Date varies in", date_array)
                    raise
            self.time_array.append(date_array[0].date())

        target_longitude = LONGITUDE_ARRAY
        target_latitude = LATITUDE_ARRAY

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
            self.model_field[key] = np.asarray(value)[
                :,latitude_min:latitude_max, longitude_min:longitude_max]
            if self.model_field[key][0].shape != target_shape:
                print(key, "does not have shape", target_shape)
                raise

        dataset_file.close()

    def get_rate(self, model_field, key):
        grad = np.gradient(model_field[key], RESOLUTION, axis=(1,2))
        return np.sqrt(
            np.square(grad[0] * model_field["y_wind"])
            + np.square(- grad[1] * model_field["x_wind"]))

    def load_rain(self, file_name):
        rain_data = netCDF4.Dataset(file_name, "r", format="NETCDF4")
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
        time_array = netCDF4.num2date(time_array[:], time_array.units)

        self.mask = rain[0].mask
        for i, time in enumerate(time_array):
            time = time.date()
            if time in self.time_array:
                self.rain.append(rain[i])
                if not np.array_equal(self.mask, rain[i].mask):
                    print("Mask in ", time, "is not consistent")
                    raise
        self.rain = ma.asarray(self.rain)

        rain_data.close()

    def load_topo(self, file_name):
        gdal_dataset = gdal.Open(file_name)
        raster_band = gdal_dataset.GetRasterBand(1)
        topo = raster_band.ReadAsArray() / GRAVITATIONAL_FIELD_STRENGTH
        grad = np.gradient(topo, RESOLUTION)
        grad = np.sqrt(np.square(grad[0]) + np.square(grad[1]))

        lat_index = []
        long_index = []
        for lat_i in LATITUDE_TOPO_ARRAY:
            lat_index.append(np.any(np.isclose(LATITUDE_ARRAY, lat_i)))
        for long_i in LONGITUDE_TOPO_ARRAY:
            long_index.append(np.any(np.isclose(LONGITUDE_ARRAY, long_i)))

        lat_index = np.where(lat_index)[0]
        long_index = np.where(long_index)[0]

        topo = topo[lat_index[0]:lat_index[-1]+1,
                    long_index[0]:long_index[-1]+1]
        grad = grad[lat_index[0]:lat_index[-1]+1,
                    long_index[0]:long_index[-1]+1]

        self.topography["elevation"] = topo
        self.topography["gradient"] = grad

        for key, value in self.topography.items():
            topo_i = value.copy()
            shift = np.mean(topo_i)
            scale = np.std(topo_i, ddof=1)
            self.topography_normalise[key] = (topo_i - shift) / scale

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
        for i, latitude in enumerate(LATITUDE_ARRAY):
            latitude_error = abs(latitude - coordinates[0])
            if min_latitude_error > latitude_error:
                latitude_index = i
                min_latitude_error = latitude_error
        #find longitude
        for i, longitude in enumerate(LONGITUDE_ARRAY):
            longitude_error = abs(longitude - coordinates[1])
            if min_longitude_error > longitude_error:
                longitude_index = i
                min_longitude_error = longitude_error
        return(latitude_index, longitude_index)

    def get_latitude_longitude_city(self, city):
        return self.find_nearest_latitude_longitude(CITY_LOCATION[city])

    def get_latitude_longitude_random(self, rng):
        latitude_index = rng.randint(0, len(LATITUDE_ARRAY))
        longitude_index = rng.randint(0, len(LONGITUDE_ARRAY))
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

    def crop(self, lat, long):
        for key, model_field in self.model_field.items():
            self.model_field[key] = model_field[
                :, lat[0]:lat[1], long[0]:long[1]]
        self.rain = self.rain[:, lat[0]:lat[1], long[0]:long[1]]
        self.mask = self.rain[0].mask
        for key in self.topography:
            self.topography[key] = self.topography[key][
                lat[0]:lat[1], long[0]:long[1]]
            self.topography_normalise[key] = self.topography_normalise[key][
                lat[0]:lat[1], long[0]:long[1]]

    def trim(self, time):
        for key, model_field in self.model_field.items():
            self.model_field[key] = model_field[time[0]:time[1], :, :]
        self.rain = self.rain[time[0]:time[1], :, :]
        self.time_array = self.time_array[time[0]:time[1]]

class DataDualGrid(Data):

    def __init__(self):
        self.model_field_interpolate = {}
        self.topography_coarse = {}
        self.topography_coarse_normalise = {}
        super().__init__()

    def copy_from(self, other):
        super().copy_from(other)
        self.model_field_interpolate = other.model_field_interpolate
        self.topography_coarse = other.topography_coarse
        self.topography_coarse_normalise = other.topography_coarse_normalise

    #override
    def get_model_field(self, latitude_index, longitude_index):
        data_frame = {}
        for model_field_name, value in self.model_field_interpolate.items():
            data_frame[model_field_name] = value[
                :, latitude_index, longitude_index]
        return pd.DataFrame(data_frame)

    def load_topo(self, file_name):
        super().load_topo(file_name)
        longitude_coarse_grid, latitude_coarse_grid = np.meshgrid(
            LONGITUDE_COARSE_ARRAY, LATITUDE_COARSE_ARRAY)
        self.topography_coarse["longitude"] = longitude_coarse_grid
        self.topography_coarse["latitude"] = latitude_coarse_grid

        for key in ["elevation", "gradient"]:
            topo_interpolate = interpolate.RectBivariateSpline(
                np.flip(LATITUDE_ARRAY), LONGITUDE_ARRAY,
                np.flip(self.topography[key], 0))
            topo_interpolate = topo_interpolate(
                np.flip(LATITUDE_COARSE_ARRAY), LONGITUDE_COARSE_ARRAY)
            self.topography_coarse[key] = np.flip(topo_interpolate, 0)

        for key in self.topography:
            topo_coarse = self.topography_coarse[key]
            topo_fine = self.topography[key]
            shift = np.mean(topo_fine)
            scale = np.std(topo_fine, ddof=1)
            self.topography_coarse_normalise[key] = (
                (topo_coarse - shift) / scale)

    def load_model_field_interpolate_to_coarse(self, file_name):
        super().load_model_field(file_name)
        self.model_field_interpolate = self.model_field.copy()
        self.model_field = {}
        for key, model_field in self.model_field_interpolate.items():
            model_field_coarse = []
            for i in range(len(self.time_array)):
                interpolator = interpolate.RectBivariateSpline(
                    np.flip(LATITUDE_ARRAY), LONGITUDE_ARRAY,
                    np.flip(model_field[i,:,:], 0))
                model_field_interpolate = interpolator(
                    np.flip(LATITUDE_COARSE_ARRAY), LONGITUDE_COARSE_ARRAY)
                model_field_interpolate = np.flip(model_field_interpolate, 0)
                model_field_coarse.append(model_field_interpolate)
            self.model_field[key] = model_field_coarse

    def interpolate_model_field(self):
        for key, model_field in self.model_field.items():
            self.model_field_interpolate[key] = []
            for i in range(len(self.time_array)):
                interpolator = interpolate.RectBivariateSpline(
                    np.flip(LATITUDE_COARSE_ARRAY), LONGITUDE_COARSE_ARRAY,
                    np.flip(model_field[i,:,:], 0))
                model_field_interpolate_i = interpolator(
                    np.flip(LATITUDE_ARRAY), LONGITUDE_ARRAY)
                model_field_interpolate_i = np.flip(
                    model_field_interpolate_i, 0)
                self.model_field_interpolate[key].append(
                    model_field_interpolate_i)
            self.model_field_interpolate[key] = np.asarray(
                self.model_field_interpolate[key])

class AnaInterpolate1(Data):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        self.load_model_field(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "ana_input_1.nc"))
        self.load_rain(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"))
        self.load_topo(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "topo_0.1_degree.grib"))

class AnaDualExample1(DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        self.load_model_field_interpolate_to_coarse(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "ana_input_1.nc"))
        self.load_rain(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"))
        self.load_topo(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "topo_0.1_degree.grib"))
