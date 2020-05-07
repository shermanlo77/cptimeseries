import os
from os import path

import joblib
import matplotlib.pyplot as plt
import numpy as np

import compound_poisson
from compound_poisson.fit import fitter

class Fitter(fitter.Fitter):

    def __init__(self, time_series_class, directory=""):
        super().__init__(time_series_class, directory)

    def initial_fit(self, dataset, seed, n_sample=None, pool=None):
        #dataset should be able to call get_data_training()
            #and get_time_training()
        model_field, rain = dataset.get_data_training()
        time_series = self.model_class(model_field, rain, (5, 5), (5, 5))
        time_series.time_array = dataset.get_time_training()
        time_series.set_rng(seed)
        time_series.memmap_dir = self.result_dir
        if not n_sample is None:
            time_series.n_sample = n_sample
        time_series.fit()
        return time_series

    def print_mcmc(self, time_series, dataset):
        #get the true parameters if it exists
        try:
            true_parameter = dataset.time_series.get_parameter_vector()
        except AttributeError:
            true_parameter = None
        #print results
        time_series.print_mcmc(self.figure_dir, true_parameter)

    def initalise_model_for_forecast(self, time_series):
        time_series.forecaster_memmap_dir = self.result_dir

    def do_forecast(self, time_series, dataset, n_simulation, pool=None):
        #dataset should be able to call get_model_field_test()
        #args pool not used
        time_series.forecast_self(n_simulation)
        time_series.forecast(dataset.get_model_field_test(), n_simulation)

    def print_forecast(self, time_series, dataset):
        #dataset should be able to call get_rain_training() and get_rain_test()
        #plot forecast results
        rain = dataset.get_rain_training()
        compound_poisson.print.forecast(
            time_series.self_forecaster, rain, self.figure_dir, "training")

        rain = dataset.get_rain_test()
        compound_poisson.print.forecast(
            time_series.forecaster, rain, self.figure_dir, "test")

class FitterMcmc(Fitter):

    def __init__(self, directory=""):
        super().__init__(compound_poisson.TimeSeriesMcmc, directory)

class FitterSlice(Fitter):

    def __init__(self, directory=""):
        super().__init__(compound_poisson.TimeSeriesSlice, directory)

class FitterHyperSlice(Fitter):

    def __init__(self, directory=""):
        super().__init__(compound_poisson.TimeSeriesHyperSlice, directory)
