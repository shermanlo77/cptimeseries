from os import path
import pathlib

import joblib


class Location(object):
    """For storing model fields and precipitation for a single location

    Attributes:
        model_field: pandas data frame, 0th dimension correspond to the time
            point
        rain: numpy array, precipitation for each time point
        time_array: array of dates
        model_field_units: dictionary of model field units
        time_series: None if real data or a time_series object if this data is
            simulated
    """

    def __init__(self):
        self.model_field = None
        self.rain = None
        self.time_array = None
        self.model_field_units = None
        self.time_series = None

        # load the .gz file if one exist, otherwise load the data and save it
        path_to_storage = pathlib.Path(__file__).parent.absolute()
        storage_file = path.join(
            path_to_storage, self.__class__.__name__+".gz")
        if path.isfile(storage_file):
            print("Loading", storage_file)
            self.copy_from(joblib.load(storage_file))
        else:
            self.load_data()
            print("Saving", storage_file)
            joblib.dump(self, storage_file)

    def copy_from(self, other):
        # required for loading .gz in constructor
        self.model_field = other.model_field
        self.rain = other.rain
        self.time_array = other.time_array
        self.model_field_units = other.model_field_units

    def load_data(self):
        # to be implemented, to load data from a data.Data object
        raise NotImplementedError

    def load_data_from_city(self, data, city):
        self.time_array = data.time_array.copy()
        model_field, rain = data.get_data_city(city)
        self.model_field = model_field.copy()
        self.rain = rain.copy()

    def get_data(self):
        return (self.get_model_field(), self.get_rain())

    def get_model_field(self):
        return self.model_field

    def get_rain(self):
        return self.rain

    def get_time(self):
        return self.time_array

    def __len__(self):
        return len(self.time_array)
