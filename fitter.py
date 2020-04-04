import os
from os import path

import joblib
import matplotlib.pyplot as plt
import numpy as np

import compound_poisson

class Fitter(object):

    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        self.time_series_class = None
        self.dataset = dataset
        self.rng = rng
        self.name = name
        self.result_dir = result_dir
        self.figure_dir = figure_dir

    def __call__(self):

        figure_sub_dir = path.join(self.figure_dir, self.name)
        if not path.isdir(figure_sub_dir):
            os.mkdir(figure_sub_dir)

        result_file = path.join(self.result_dir, self.name + ".gz")
        if not path.isfile(result_file):
            model_field, rain = self.dataset.get_data_training()
            time_series = self.time_series_class(
                model_field, rain, (5, 5), (5, 5))
            time_series.time_array = self.dataset.get_time_training()
            time_series.rng = self.rng
            time_series.fit()
            joblib.dump(time_series, result_file)
        else:
            time_series = joblib.load(result_file)

        try:
            true_parameter = self.dataset.time_series.get_parameter_vector()
        except AttributeError:
            true_parameter = None
        directory = path.join(self.figure_dir, self.name)
        time_series.print_mcmc(directory, true_parameter)

class FitterMcmc(Fitter):

    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        super().__init__(dataset, rng, name, result_dir, figure_dir)
        self.time_series_class = compound_poisson.TimeSeriesMcmc


class FitterSlice(Fitter):

    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        super().__init__(dataset, rng, name, result_dir, figure_dir)
        self.time_series_class = compound_poisson.TimeSeriesSlice

class FitterHyperSlice(FitterSlice):

    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        super().__init__(dataset, rng, name, result_dir, figure_dir)
        self.time_series_class = compound_poisson.TimeSeriesHyperSlice
