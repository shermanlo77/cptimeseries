import os
from os import path

import compound_poisson
from compound_poisson.fit import fitter
from compound_poisson import forecast


class Fitter(fitter.Fitter):

    def __init__(self, time_series_class, directory="", suffix=None):
        super().__init__(time_series_class, directory, suffix)

    def initial_fit(self, dataset, seed, n_sample=None, pool=None):
        # dataset contains training set
        model_field, rain = dataset.get_data()
        time_series = self.model_class(model_field, rain, (5, 5), (5, 5))
        time_series.time_array = dataset.get_time()
        time_series.set_rng(seed)
        time_series.memmap_dir = self.result_dir
        if n_sample is not None:
            time_series.n_sample = n_sample
        time_series.fit()
        return time_series

    def print_mcmc(self, time_series, dataset, pool=None):
        # get the true parameters if it exists
        # pool not used
        if dataset.time_series is None:
            true_parameter = None
        else:
            true_parameter = dataset.time_series.get_parameter_vector()
        # print results
        directory = path.join(self.figure_dir, "chain")
        if not path.isdir(directory):
            os.mkdir(directory)
        time_series.print_mcmc(directory, true_parameter)

    def initalise_model_for_forecast(self, time_series):
        time_series.forecaster_memmap_dir = self.result_dir

    def do_forecast(self, time_series, dataset, n_simulation, pool=None):
        # dataset contains (training set, test set)
        # only the test set is needed by the two are passed because these are
        # passed in the super class method forecast. The subclasses of the
        # super class includes downscale.py which doesn't require the require
        # training set (which may change in future development)
        time_series.forecast_self(n_simulation)
        time_series.forecast(dataset[1].get_model_field(), n_simulation)

    def print_forecast(self, time_series, dataset, pool=None):
        # dataset contains (training set, test set)
        rain = dataset[0].get_rain()
        printer = forecast.print.TimeSeries(
            time_series.self_forecaster, rain, self.figure_dir, "training_")
        printer.print()

        rain = dataset[1].get_rain()
        printer = forecast.print.TimeSeries(
            time_series.forecaster, rain, self.figure_dir, "test_")
        printer.print()


class FitterMcmc(Fitter):

    def __init__(self, directory="", suffix=None):
        super().__init__(compound_poisson.TimeSeriesMcmc, directory, suffix)


class FitterSlice(Fitter):

    def __init__(self, directory="", suffix=None):
        super().__init__(compound_poisson.TimeSeriesSlice, directory, suffix)


class FitterHyperSlice(Fitter):

    def __init__(self, directory="", suffix=None):
        super().__init__(
            compound_poisson.TimeSeriesHyperSlice, directory, suffix)
