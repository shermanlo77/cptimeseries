import os

import numpy as np
from numpy import random

import compound_poisson as cp
import dataset

class London80(object):
    
    def __init__(self):
        self.training_range = range(0, 3653)
        self.test_range = range(3653, 4018)
        self.model_field = None
        self.rain = None
        self.time_array = None
        self.load_data()
    
    def load_data(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data = dataset.Ana_1()
        self.time_array = data.time_array.copy()
        model_field, rain = data.get_data_city("London")
        self.model_field = model_field.copy()
        self.rain = rain.copy()
    
    def get_data_training(self):
        return self.get_model_field_training(), self.get_rain_training()
    
    def get_data_test(self):
        return self.get_model_field_test(), self.get_rain_test()
    
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
    
class LondonSimulated80(London80):
    
    def __init__(self):
        self.time_series = None
        super().__init__()
    
    def load_data(self):
        super().load_data()
        rng = random.RandomState(np.uint32(3667413888))
        n_model_field = len(self.model_field.columns)
        n_arma = (5, 5)
        poisson_rate = cp.PoissonRate(n_model_field, n_arma)
        gamma_mean = cp.GammaMean(n_model_field, n_arma)
        gamma_dispersion = cp.GammaDispersion(n_model_field)
        cp_parameter_array = [poisson_rate, gamma_mean, gamma_dispersion]
        poisson_rate['reg'] = np.asarray([
            -0.11154721,
            0.01634086,
            0.45595715,
            -0.3993777,
            0.09398412,
            -0.22794538,
            0.07249126,
            -0.21600272,
            -0.05372614,
        ])
        poisson_rate['const'] = np.asarray([-0.92252178])
        poisson_rate['AR'] = np.asarray([
            0.2191157,
            0.0828164,
            -0.08994476,
            0.08133209,
            -0.09344768,
        ])
        poisson_rate['MA'] = np.asarray([
            0.22857258,
            0.16147521,
            -0.02136632,
            0.04896173,
            0.02372191,
        ])
        gamma_mean['reg'] = np.asarray([
            -0.09376735,
            -0.01028988, 
            0.02133337,
            0.15878673,
            -0.15329763,
            0.17121309,
            -0.18262059,
            -0.1709044,
            -0.11908832,
        ])
        gamma_mean['const'] = np.asarray([1.18446041])
        gamma_mean['AR'] = np.asarray([
            0.22679054,
            0.10105583,
            -0.05324423,
            0.03245928,
            0.02608218,
        ])
        gamma_mean['MA'] = np.asarray([
            0.21196802,
            0.18057783,
            -0.06592883,
            0.06715984,
            -0.05437931,
        ])
        gamma_dispersion['reg'] = np.asarray([
            0.07291021,
            0.34183881,
            0.20085349,
            0.2210854,
            0.1586696,
            0.37656874,
            0.03970588,
            -0.15201423,
            -0.13569733,
        ])
        gamma_dispersion['const'] = np.asarray([-0.26056112])
        
        self.time_series = cp.TimeSeries(
            self.model_field, cp_parameter_array=cp_parameter_array)
        self.time_series.rng = rng
        self.time_series.simulate()
        self.time_series.time_array = self.time_array
        self.rain = self.time_series.y_array