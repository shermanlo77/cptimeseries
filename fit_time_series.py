import os
from os import path

import joblib
import matplotlib.pyplot as plt
import numpy as np

import compound_poisson

class Fitter(object):
    """Wrapper class for fitting TimeSeries onto data

    Attributes:
        name: name of TimeSeries object to be fitted
        time_series_class: TimeSeries class to use
        result_dir: directory to store result files
        figure_dir: directory to store figures
    """

    def __init__(self, directory, time_series_class):
        """
        Args:
            directory: location to store results and figures
            time_series_class: TimeSeries class to use
        """
        self.name = time_series_class.__name__
        self.time_series_class = time_series_class
        self.result_dir = path.join(directory, "result")
        self.figure_dir = path.join(directory, "figure")

        if not path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        if not path.isdir(self.figure_dir):
            os.mkdir(self.figure_dir)

        self.figure_dir = path.join(self.figure_dir, self.name)
        if not path.isdir(self.figure_dir):
            os.mkdir(self.figure_dir)

    def fit(self, dataset, seed, n_sample=None):
        """Fit model onto data

        Call to do MCMC, call again to resume. Reproducible if forecast() is not
            called between fit() calls. The argument n_sample is designed for
            debugging purposes.

        Args:
            dataset: object with method get_data_training() which returns model
                fields and precipitation
            seed: numpy.random.SeedSequence object
            n_sample: number of samples, default value used is None is passed
        """
        result_file = path.join(self.result_dir, self.name + ".gz")

        #check if this is the initial fit() call by checking the results files
        if not path.isfile(result_file):
            #fit the model
            model_field, rain = dataset.get_data_training()
            time_series = self.time_series_class(
                model_field, rain, (5, 5), (5, 5))
            time_series.time_array = dataset.get_time_training()
            time_series.set_rng(seed)
            time_series.memmap_path = self.result_dir
            if not n_sample is None:
                time_series.n_sample = n_sample
            time_series.fit()
            #save results
            joblib.dump(time_series, result_file)
        #else this is not an initial call, resume the MCMC fit
        else:
            time_series = joblib.load(result_file)
            if not n_sample is None:
                time_series.resume_fitting(n_sample)
                joblib.dump(time_series, result_file)

        #get the true parameters if it exists
        try:
            true_parameter = dataset.time_series.get_parameter_vector()
        except AttributeError:
            true_parameter = None

        #print results
        time_series.read_memmap()
        time_series.print_mcmc(self.figure_dir, true_parameter)

    def forecast(self, dataset, n_simulation=None, burn_in=None):
        """Make forecasts on training and test set

        After fitting, make forecasts on training and test set. When called
            again, can add more simulations.
        NOTE: THIS IS NOT REPRODUCIBLE IF CALLED AGAIN (this is because it does n_simulation)

        Args:
            dataset: object which can call get_model_field_test() and
                get_rain_test()
            n_simulation: number of simulations, default is 1000
            burn_in: burn in for MCMC, default is 1000
        """
        result_file = path.join(self.result_dir, self.name + ".gz")
        time_series = joblib.load(result_file)
        if n_simulation is None:
            n_simulation = 1000
        if not burn_in is None:
            time_series.burn_in = burn_in
        time_series.forecaster_memmap_dir = self.result_dir

        time_series.forecast_self(n_simulation)
        time_series.forecast(dataset.get_model_field_test(), n_simulation)
        joblib.dump(time_series, result_file)

        rain = dataset.get_rain_training()
        compound_poisson.print.forecast(
            time_series.self_forecaster, rain, self.figure_dir, "training")

        rain = dataset.get_rain_test()
        compound_poisson.print.forecast(
            time_series.forecaster, rain, self.figure_dir, "test")

class FitterMcmc(Fitter):

    def __init__(self, directory):
        super().__init__(directory, compound_poisson.TimeSeriesMcmc)

class FitterSlice(Fitter):

    def __init__(self, directory):
        super().__init__(directory, compound_poisson.TimeSeriesSlice)

class FitterHyperSlice(Fitter):

    def __init__(self, directory):
        super().__init__(directory, compound_poisson.TimeSeriesHyperSlice)
