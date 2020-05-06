import os
from os import path

import joblib
import matplotlib.pyplot as plt
import numpy as np

import compound_poisson

class Fitter(object):
    """Wrapper class for fitting Downscale onto data and check-pointing

    Attributes:
        name: name of Downscale object to be fitted
        downscale_class: Downscale class to use
        result_dir: directory to store result files
        figure_dir: directory to store figures
    """

    def __init__(self, directory, downscale_class):
        self.name = downscale_class.__name__
        self.downscale_class = downscale_class
        self.result_dir = path.join(directory, "result")
        self.figure_dir = path.join(directory, "figure")

        if not path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        if not path.isdir(self.figure_dir):
            os.mkdir(self.figure_dir)

        self.figure_dir = path.join(self.figure_dir, self.name)
        if not path.isdir(self.figure_dir):
            os.mkdir(self.figure_dir)

    def fit(self, dataset, seed, pool=None, n_sample=None):
        """Fit model onto data

        Call to do MCMC, call again to resume. Reproducible if forecast() is not
            called between fit() calls. The argument n_sample is designed for
            debugging purposes.

        Args:
            dataset: training set, Data object
            seed: numpy.random.SeedSequence object
            pool: compound_poisson.multiprocess object
            n_sample: number of samples, default value used is None is passed
        """
        result_file = path.join(self.result_dir, self.name + ".gz")
        #check if this is the initial fit() call by checking the results files
        if not path.isfile(result_file):
            downscale = self.downscale_class(dataset, (5, 5))
            downscale.set_rng(seed)
            downscale.set_memmap_path(self.result_dir)
            if not n_sample is None:
                downscale.n_sample = n_sample
            downscale.fit(pool)
            joblib.dump(downscale, result_file)
        #else this is not an initial call, resume the MCMC fit
        else:
            downscale = joblib.load(result_file)
            if not n_sample is None:
                downscale.resume_fitting(n_sample, pool)
                joblib.dump(downscale, result_file)

        downscale.print_mcmc(self.figure_dir)

    def forecast(self, dataset, pool, n_simulation, burn_in=0):
        """Make forecasts on test set

        After fitting, make forecasts on test set. When called again, can add
            more simulations. The method fit() sould not be called after calling
            forecast(). Recommend backing up the results before calling
            forecast() when debugging or developing.

        Args:
            dataset: test set, Data object
            pool: compound_poisson.multiprocess object
            n_simulation: number of simulations, default is 1000
            burn_in: burn in for MCMC, default value is 0
        """
        result_file = path.join(self.result_dir, self.name + ".gz")
        downscale = joblib.load(result_file)
        #set default n_simulation if this is the initial forecast() call
        if (downscale.forecaster is None) and (n_simulation is None):
            n_simulation = 1000
        if not burn_in is None:
            downscale.set_burn_in(burn_in)

        #call downscale.forecast() if this is the initial forecast() call
            #or this is further forecast() calls and a n_simulation is provided
        is_do_forecast = ((downscale.forecaster is None)
            or ((not downscale.forecaster is None)
            and (not n_simulation is None)))
        if is_do_forecast:
            downscale.forecast(dataset, n_simulation, pool)
            joblib.dump(downscale, result_file)

        compound_poisson.print.downscale_forecast(
            downscale.forecaster, dataset, self.figure_dir)

class FitterDownscale(Fitter):

    def __init__(self, directory):
        super().__init__(directory, compound_poisson.Downscale)

class FitterDownscaleDual(Fitter):

    def __init__(self, directory):
        super().__init__(directory, compound_poisson.DownscaleDual)
