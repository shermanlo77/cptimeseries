import os
from os import path

import joblib
import matplotlib.pyplot as plt
import numpy as np

import compound_poisson

class Fitter(object):

    def __init__(self, name, directory, rng, time_series_class):
        self.name = name
        self.rng = rng
        self.time_series_class = time_series_class
        self.result_dir = path.join(directory, "result")
        self.figure_dir = path.join(directory, "figure")

        if not path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        if not path.isdir(self.figure_dir):
            os.mkdir(self.figure_dir)

    def fit(self, dataset, n_sample=None):
        figure_sub_dir = path.join(self.figure_dir, self.name)
        if not path.isdir(figure_sub_dir):
            os.mkdir(figure_sub_dir)

        result_file = path.join(self.result_dir, self.name + ".gz")
        if not path.isfile(result_file):
            model_field, rain = dataset.get_data_training()
            time_series = self.time_series_class(
                model_field, rain, (5, 5), (5, 5))
            time_series.time_array = dataset.get_time_training()
            time_series.rng = self.rng
            time_series.memmap_path = self.result_dir
            if not n_sample is None:
                time_series.n_sample = n_sample
            time_series.fit()
            joblib.dump(time_series, result_file)
        else:
            time_series = joblib.load(result_file)
            if not n_sample is None:
                time_series.resume(n_sample)
                joblib.dump(time_series, result_file)
        try:
            true_parameter = dataset.time_series.get_parameter_vector()
        except AttributeError:
            true_parameter = None
        directory = path.join(self.figure_dir, self.name)
        time_series.read_memmap()
        time_series.print_mcmc(directory, true_parameter)

class FitterMcmc(Fitter):

    def __init__(self, name, directory, rng):
        super().__init__(name, directory, rng, compound_poisson.TimeSeriesMcmc)

class FitterSlice(Fitter):

    def __init__(self, name, directory, rng):
        super().__init__(name, directory, rng, compound_poisson.TimeSeriesSlice)

class FitterHyperSlice(Fitter):

    def __init__(self, name, directory, rng):
        super().__init__(
             name, directory, rng, compound_poisson.TimeSeriesHyperSlice)
