import os

from .Data import Data
import joblib

class London80:
    
    def __init__(self):
        self.training_range = range(0, 3653)
        self.test_range = range(3653, 4018)
        self.model_field = None
        self.rain = None
        self.time_array = None
        self.load_data()
    
    def load_data(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data = joblib.load(os.path.join(dir_path, "ana_input_1.gz"))
        self.time_array = data.time_array.copy()
        model_field, rain = data.get_data_city("London")
        self.model_field = model_field.copy()
        self.rain = rain.copy()
    
    def get_data_training(self):
        return (self.get_model_field_training(), self.get_rain_training())
    
    def get_data_test(self):
        return (self.get_model_field_test(), self.get_rain_test())
    
    def get_model_field_training(self):
        return self.model_field[
            self.training_range.start : self.training_range.stop]
    
    def get_model_field_test(self):
        return self.model_field[self.test_range.start : self.test_range.stop]
    
    def get_rain_training(self):
        return self.rain[self.training_range.start : self.training_range.stop]
    
    def get_rain_test(self):
        return self.rain[self.test_range.start : self.test_range.stop]
    
    def get_time_training(self):
        return self.time_array[
            self.training_range.start : self.training_range.stop]
    
    def get_time_test(self):
        return self.time_array[self.test_range.start : self.test_range.stop]
    
    
