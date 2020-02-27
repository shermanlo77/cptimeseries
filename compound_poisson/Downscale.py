import numpy as np

from .TimeSeriesSlice import TimeSeriesSlice

import pdb

class Downscale:
    
    def __init__(self, data, n_arma=(0,0)):
        self.n_arma = n_arma
        self.time_series_array = []
        self.mask = data.mask
        self.parameter_mask_vector = []
        self.n_parameter = None
        self.topography_array = data.topography
        self.shape = self.mask.shape
        self.area = self.shape[0] * self.shape[1]
        self.area_unmask = np.sum(np.logical_not(self.mask))
        
        model_field = data.model_field
        rain = data.rain
        time_series_array = self.time_series_array
        
        for lat_i in range(self.shape[0]):
            
            time_series_array.append([])
            
            for long_i in range(self.shape[1]):
                
                x_i, rain_i = data.get_data(lat_i, long_i)
                is_mask = self.mask[lat_i, long_i]
                if is_mask:
                    time_series = TimeSeriesSlice(x_i,
                                                  poisson_rate_n_arma=n_arma,
                                                  gamma_mean_n_arma=n_arma)
                else:
                    time_series = TimeSeriesSlice(x_i, rain_i, n_arma, n_arma)
                time_series.id = str(lat_i) + "_" + str(long_i)
                time_series_array[lat_i].append(time_series)
                for i in range(time_series.n_parameter):
                    self.parameter_mask_vector.append(is_mask)
                self.n_parameter = time_series.n_parameter
        
        self.parameter_mask_vector = np.asarray(self.parameter_mask_vector)
    
    def get_parameter_3d(self):
        parameter = []
        for time_series_lat in self.time_series_array:
            parameter.append([])
            for time_series_i in time_series_lat:
                parameter[-1].append(time_series_i.get_parameter_vector())
        parameter = np.asarray(parameter)
        return parameter
    
    def get_parameter_vector_mask(self, is_mask):
        parameter_3d = self.get_parameter_3d()
        parameter_vector = []
        for i in range(self.n_parameter):
            if is_mask:
                parameter_vector.append(
                    parameter_3d[np.logical_not(self.mask), i].flatten())
            else:
                parameter_vector.append(parameter_3d[:, :, i].flatten())
        return np.asarray(parameter_vector).flatten()
    
    def get_parameter_vector(self):
        return self.get_parameter_vector_mask(True)
    
    def get_parameter_vector_all(self):
        return self.get_parameter_vector_mask(False)
    
    def set_parameter_vector_mask(self, parameter_vector, is_mask):
        counter = 0
        if is_mask:
            parameter_step = self.area_unmask
        else:
            parameter_step = self.area
        parameter_3d = np.empty(
            (self.shape[0], self.shape[1], self.n_parameter))
        for i in range(self.n_parameter):
            parameter_i = parameter_vector[
                i*parameter_step : (i+1)*parameter_step]
            if is_mask:
                parameter_3d[np.logical_not(self.mask), i] = parameter_i
            else:
                parameter_3d[:,:,i] = np.reshape(parameter_i, self.shape)
        for lat_i in range(self.shape[0]):
            for long_i in range(self.shape[1]):
                if not (is_mask and self.mask[lat_i, long_i]):
                    parameter_i = parameter_3d[lat_i, long_i, :]
                    self.time_series_array[lat_i][long_i].set_parameter_vector(
                        parameter_i)
                
    def set_parameter_vector(self, parameter_vector):
        self.set_parameter_vector_mask(parameter_vector, True)
    
    def set_parameter_vector_all(self, parameter_vector):
        self.set_parameter_vector_mask(parameter_vector, False)
    
    def get_parameter_vector_name_3d(self):
        parameter_name = []
        for time_series_lat in self.time_series_array:
            parameter_name.append([])
            for time_series_i in time_series_lat:
                parameter_name[-1].append(
                    time_series_i.get_parameter_vector_name())
        parameter_name = np.asarray(parameter_name)
        return parameter_name
    
    def get_parameter_vector_name_mask(self, is_mask):
        parameter_name_3d = self.get_parameter_vector_name_3d()
        parameter_name_array = []
        for i in range(self.n_parameter):
            if is_mask:
                parameter_name_array.append(
                    parameter_name_3d[np.logical_not(self.mask), i].flatten())
            else:
                parameter_name_array.append(
                    parameter_name_3d[:, :, i].flatten())
        return np.asarray(parameter_name_array).flatten()
    
    def get_parameter_vector_name(self):
        return self.get_parameter_vector_name_mask(True)
    
    def get_parameter_vector_name_all(self):
        return self.get_parameter_vector_name_mask(False)
