import os
from os import path

import joblib

class Fitter(object):
    """Wrapper class for fitting models onto data and check-pointing

    Attributes:
        name: name of class for an instantiated object to be fitted
        model_class: class to use
        result_dir: directory to store result files
        figure_dir: directory to store figures
    """

    def __init__(self, model_class, directory):
        """
        Args:
            model_class: class to use
            directory: location to store results and figures
        """
        self.name = model_class.__name__
        self.model_class = model_class
        self.result_dir = path.join(directory, "result")
        self.figure_dir = path.join(directory, "figure")

        if not path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        if not path.isdir(self.figure_dir):
            os.mkdir(self.figure_dir)

        self.figure_dir = path.join(self.figure_dir, self.name)
        if not path.isdir(self.figure_dir):
            os.mkdir(self.figure_dir)

    def get_result_path(self):
        """Return path where the model is to be stored
        """
        return path.join(self.result_dir, self.name + ".gz")

    def fit(self, dataset, seed, n_sample=None, pool=None):
        """Fit model onto data

        Call to do MCMC, call again to resume. Reproducible if forecast() is not
            called between fit() calls. The argument n_sample is designed for
            debugging purposes.

        Args:
            dataset: object containing training (and/or test) data (exact class
                varies)
            seed: numpy.random.SeedSequence object
            n_sample: number of samples, default value used if None is passed
            pool: required by downscale
        """
        result_file = self.get_result_path()
        #check if this is the initial fit() call by checking the results files
        if not path.isfile(result_file):
            #fit the model
            model = self.initial_fit(dataset, seed, n_sample, pool)
            #save results
            joblib.dump(model, result_file)
        #else this is not an initial call, resume the MCMC fit
        else:
            model = joblib.load(result_file)
            if not n_sample is None:
                if pool is None:
                    model.resume_fitting(n_sample)
                else:
                    model.resume_fitting(n_sample, pool)
                joblib.dump(model, result_file)
        self.print_mcmc(model, dataset)

    def initial_fit(self, dataset, seed, n_sample=None, pool=None):
        """Initial fit

        Return a fitted model

        Args:
            dataset: object containing training data (exact class varies)
            seed: numpy.random.SeedSequence object
            n_sample: number of samples, default value used if None is passed
            pool: required by downscale
        """
        #pool is required by downscale
        raise NotImplementedError

    def print_mcmc(self, model, dataset=None):
        """Print mcmc chain of the fitted model
        """
        #dataset is required by time_series for plotting true values
        raise NotImplementedError

    def forecast(self,
                 dataset,
                 n_simulation=None,
                 burn_in=None,
                 pool=None,
                 is_print=True):
        """Make forecasts on training and test set

        After fitting, make forecasts on test (and training) set. When called
            again, can add more simulations. The method fit() should not be
            called after calling forecast(). Recommend backing up the results
            before calling forecast() when debugging or developing.

        Args:
            dataset: object containing test data (exact class varies)
            n_simulation: number of simulations, default is 1000
            burn_in: burn in for MCMC, default value is 0
            pool: required by downscale
            is_print: boolean, print forecasts if True
        """
        result_file = self.get_result_path()
        model = joblib.load(result_file)
        self.initalise_model_for_forecast(model)

        #set default n_simulation if this is the initial forecast() call
        if (model.forecaster is None) and (n_simulation is None):
            n_simulation = 1000
        if not burn_in is None:
            model.set_burn_in(burn_in)

        #call do_forecast() if this is the initial do_forecast() call or this is
            #further do_forecast() calls and a n_simulation is provided
        is_do_forecast = ((model.forecaster is None)
            or ((not model.forecaster is None)
            and (not n_simulation is None)))
        if is_do_forecast:
            self.do_forecast(model, dataset, n_simulation, pool)
            joblib.dump(model, result_file)

        if is_print:
            self.print_forecast(model, dataset, pool)

    def initalise_model_for_forecast(self, model):
        """Initalise the model object right after loading
        """
        pass

    def do_forecast(self, model, dataset, n_simulation, pool=None):
        """Make forecasts for the model
        """
        #pool required by downscale
        raise NotImplementedError

    def print_forecast(self, model, dataset, pool=None):
        """Print forecast figures for this model
        """
        #pool required by downscale
        raise NotImplementedError
