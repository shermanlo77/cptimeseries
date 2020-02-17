import numpy as np
import pandas as pd
import math
import compound_poisson as cp
from netCDF4 import Dataset, num2date
from TimeSeries import TimeSeries
import numpy.random as random

def find_nearest_latitude_longitude(
    coordinates, latitude_array, longitude_array):
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

class Location:
    
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.training_range = range(0, 3653)
        self.test_range = range(3653, 4018)
        self.model_field_file = "../Data/Rain_Data_Nov19/ana_input_1.nc"
        self.rain_file = "../Data/Rain_Data_Nov19/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"
    
    def get_time_series_training(self):
        time_series = cp.TimeSeries(
            self.get_model_field_training(), self.get_rain_training())
        time_series.time_array = self.get_time_training()
        return time_series
    
    def get_time_series_test(self):
        time_series = cp.TimeSeries(
            self.get_model_field_test(), self.get_rain_test())
        time_series.time_array = self.get_time_test()
        return time_series
    
    def get_rain(self):
        rain_data = Dataset(self.rain_file, "r", format="NETCDF4")
        rain_data = rain_data.variables
        #get the longitude and latitude
        longitude_array = np.round_(rain_data["longitude"], 2)
        latitude_array = np.round_(rain_data["latitude"], 2)
        #for plotting, get a grid of longitudes and latitudes
        longitude_grid, latitude_grid = np.meshgrid(
            longitude_array, latitude_array)

        #get coordinates for this city (select a specific longitude and latitude)
        latitude_index, longitude_index = find_nearest_latitude_longitude(
            self.coordinates, latitude_array, longitude_array)
        #get the time series for this city
        rainfall = rain_data["rr"][:,latitude_index,longitude_index]
        rainfall = np.asarray(rainfall)
        return rainfall[self.training_range[0]:self.test_range[1]]

    def get_rain_training(self):
        rainfall = self.get_rain()
        return rainfall[self.training_range]

    def get_rain_test(self):
        rainfall = self.get_rain()
        return rainfall[self.test_range]

    def get_time(self):
        model_fields_data = Dataset(self.rain_file, "r", format="NETCDF4")
        model_fields_data = model_fields_data.variables
        time = model_fields_data["time"]
        time = num2date(np.asarray(time), time.units)
        return time

    def get_time_training(self):
        time = self.get_time()
        return time[self.training_range]

    def get_time_test(self):
        time = self.get_time()
        return time[self.test_range]

    def get_model_field(self):
        #get model fields data via netcdf4
        model_fields_data = Dataset(
            self.model_field_file, "r", format="NETCDF4")
        model_fields_data = model_fields_data.variables

        #store data, key: variable name, value: array of values for each time
        data_frame = {}
        #in model fields, 4 readings per day, take the mean over a day
        n = model_fields_data["time"].size

        #get the longitude and latitude in the model fields grid
        longitude_array = np.asarray(model_fields_data["longitude"])
        latitude_array = np.asarray(model_fields_data["latitude"])
        #get coordinates for this city (select a specific longitude and
            #latitude)
        latitude_index, longitude_index = find_nearest_latitude_longitude(
            self.coordinates, latitude_array, longitude_array)

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
                model_field_series = model_field[
                    :,latitude_index,longitude_index]
                
                model_field_array = self.combine_daily_model_field(
                    model_field_series)
                
                for i in range(len(model_field_array)):
                    data_frame[model_field_name] = model_field_array[i]
                    
        x = pd.DataFrame(data_frame)
        return x
    
    def combine_daily_model_field(self, model_field_series):
        model_field_average = []
        for i in range(0, len(model_field_series), 4):
            model_field_average.append(np.mean(model_field_series[i:i+4]))
        return [model_field_average]
    
    def get_model_field_training(self):
        model_field = self.get_model_field()
        return model_field.iloc[self.training_range]

    def get_model_field_test(self):
        model_field = self.get_model_field()
        return model_field.iloc[self.test_range]

class LocationSimulation(Location):
    
    def __init__(self, coordinates, seed):
        super().__init__(coordinates)
        self.cp_parameter_array = None
        self.seed = seed
        self.time_series_training = None
        self.time_series_test = None
    
    def get_time_series_training(self):
        self.get_rain()
        return self.time_series_training
    
    def get_time_series_test(self):
        self.get_rain()
        return self.time_series_test
    
    def get_rain(self):
        model_field = self.get_model_field_training()
        self.time_series_training = TimeSeries(
            np.asarray(model_field),
            cp_parameter_array=self.cp_parameter_array)
        self.time_series_training.rng = random.RandomState(self.seed)
        self.time_series_training.model_field_name = model_field.columns
        self.time_series_training.time_array = self.get_time_training()
        self.time_series_training.simulate()
        
        model_field = self.get_model_field_test()
        self.time_series_test = self.time_series_training.simulate_future(
            np.asarray(model_field))
        self.time_series_test.model_field_name = model_field.columns
        self.time_series_test.time_array = self.get_time_test()
        
        return np.concatenate(
            (self.time_series_training.y_array, self.time_series_test.y_array))

class RandomLocation(Location):
    def __init__(self, rng):
        self.rng = rng
        self.latitude_array = np.linspace(48.95, 59.15, 103)
        self.longitude_array = np.linspace(-11.15, 3.15, 144)
        latitude = self.rng.choice(self.latitude_array)
        longitude = self.rng.choice(self.longitude_array)
        super().__init__([latitude, longitude])
    
    def set_new_location(self):
        self.coordinates = [
            self.rng.choice(self.latitude_array),
            self.rng.choice(self.longitude_array),
        ]
    
class London(Location):
    def __init__(self):
        super().__init__([51.5074, -0.1278])

class LondonSimulation(LocationSimulation):
    
    def __init__(self):
        super().__init__([51.5074, -0.1278], np.uint32(3667413888))
        n_model_field = 6
        n_arma = (5, 10)
        poisson_rate = cp.PoissonRate(n_model_field, n_arma)
        gamma_mean = cp.GammaMean(n_model_field, n_arma)
        gamma_dispersion = cp.GammaDispersion(n_model_field)
        self.cp_parameter_array = [poisson_rate, gamma_mean, gamma_dispersion]
        poisson_rate['reg'] = np.asarray([
            -0.11154721,
            0.01634086,
            0.45595715,
            -0.3993777,
            0.09398412,
            -0.22794538,
        ])
        poisson_rate['const'] = np.asarray([-0.92252178])
        poisson_rate['AR'] = np.asarray([
            0.0828164 , -0.08994476,  0.08133209, -0.09344768,  0.2191157
        ])
        poisson_rate['MA'] = np.asarray([
            0.22857258, -0.02136632,  0.26147521,  0.14896173,  0.02372191,
        0.07249126, -0.21600272, -0.05372614, -0.18262059, -0.1709044
        ])
        gamma_mean['reg'] = np.asarray([
            -0.09376735, -0.01028988,  0.02133337,  0.15878673, -0.15329763,
        0.17121309
        ])
        gamma_mean['const'] = np.asarray([1.18446041])
        gamma_mean['AR'] = np.asarray([
            0.13245928,  0.42679054, -0.25324423,  0.40105583,  0.12608218
        ])
        gamma_mean['MA'] = np.asarray([
            -0.26592883,  0.06715984, -0.05437931, -0.51196802,  0.38057783,
       -0.11908832,  0.03970588, -0.15201423, -0.13569733,  0.19229491
        ])
        gamma_dispersion['reg'] = np.asarray([
            0.07291021,
            0.34183881,
            0.20085349,
            0.2210854,
            0.1586696,
            0.37656874,
        ])
        gamma_mean['const'] = np.asarray([0.26056112])
        
