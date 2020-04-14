import os
from os import path

import joblib
import matplotlib.pyplot as plt
import numpy as np

import compound_poisson

class Fitter(object):

    def __init__(self, name, directory, downscale_class, pool=None, seed=None):
        self.name = name
        self.seed = seed
        self.pool = pool
        self.downscale_class = downscale_class
        self.result_dir = path.join(directory, "result")
        self.figure_dir = path.join(directory, "figure")

        if not path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        if not path.isdir(self.figure_dir):
            os.mkdir(self.figure_dir)

    def fit(self, dataset):
        figure_sub_dir = path.join(self.figure_dir, self.name)
        if not path.isdir(figure_sub_dir):
            os.mkdir(figure_sub_dir)

        result_file = path.join(self.result_dir, self.name + ".gz")
        if not path.isfile(result_file):
            downscale = self.downscale_class(dataset, (5, 5))
            downscale.set_rng(self.seed)
            downscale.set_memmap_path(self.result_dir)
            downscale.fit(self.pool)
            joblib.dump(downscale, result_file)
        else:
            downscale = joblib.load(result_file)

        directory = path.join(self.figure_dir, self.name)
        downscale.print_mcmc(directory)

    def forecast(self, dataset, n_simulation, burn_in=0):
        result_file = path.join(self.result_dir, self.name + ".gz")
        downscale = joblib.load(result_file)
        forecast_file = path.join(self.result_dir, self.name + "_forecast.gz")
        if not path.isfile(forecast_file):
            downscale.burn_in = burn_in
            forecast = downscale.forecast(dataset, n_simulation, self.pool)
            joblib.dump(forecast, forecast_file)
        else:
            forecast = joblib.load(forecast_file)
        figure_sub_dir = path.join(self.figure_dir, self.name)
        compound_poisson.print.downscale_forecast(
            forecast, dataset, figure_sub_dir)

class FitterDownscale(Fitter):

    def __init__(self, name, directory, pool=None, seed=None):
        super().__init__(
            name, directory, compound_poisson.Downscale, pool, seed)

class FitterDownscaleDual(Fitter):

    def __init__(self, name, directory, pool=None, seed=None):
        super().__init__(
            name, directory, compound_poisson.DownscaleDual, pool, seed)
