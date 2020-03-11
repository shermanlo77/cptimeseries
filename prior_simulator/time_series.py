import math
import os
from os import path

import cycler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa import stattools

import compound_poisson
import dataset

class PriorSimulator(object):
    
    def __init__(self, figure_directory, rng):
        self.figure_directory = figure_directory
        self.rng = rng
        self.n_arma = (5, 5)
        self.n_simulate = 100
        self.n_lag = 20
        self.max_rain = 100
        self.data = dataset.Ana_1()
    
    def simulate_prior_time_series(self, x, prior_std):
        time_series = compound_poisson.TimeSeriesMcmc(
            x, poisson_rate_n_arma=self.n_arma,
            gamma_mean_n_arma=self.n_arma)
        time_series.rng = self.rng
        time_series.simulate_from_prior()
        return time_series
    
    def simulate_hyper_time_series(self, x):
        time_series = compound_poisson.TimeSeriesHyperSlice(
            x, poisson_rate_n_arma=self.n_arma,
            gamma_mean_n_arma=self.n_arma)
        time_series.rng = self.rng
        time_series.simulate_from_prior()
        return time_series
    
    def get_time_series(self, prior_std=None):
        x = self.data.get_model_field_random(self.rng)
        if prior_std is None:
            time_series = self.simulate_hyper_time_series(x)
        else:
            time_series = self.simulate_prior_time_series(x, prior_std)
        time_series.time_array = self.data.time_array
        return time_series
    
    def print(self, figure_directory=None, prior_std=None):
        
        if figure_directory is None:
            figure_directory = self.figure_directory
        if not path.isdir(figure_directory):
            os.mkdir(figure_directory)
        
        acf_array = []
        pacf_array = []
        rain_sorted_array = []
        
        for i_simulate in range(self.n_simulate):
            time_series = self.get_time_series(prior_std)
            compound_poisson.print.time_series(
                time_series, figure_directory, str(i_simulate) + "_")
            y = time_series.y_array
            acf = stattools.acf(y, nlags=self.n_lag, fft=True)
            try:
                pacf = stattools.pacf(y, nlags=self.n_lag)
            except(stattools.LinAlgError):
                pacf = np.full(self.n_lag + 1, np.nan)
            if (len(acf) == self.n_lag + 1) and (not np.any(np.isnan(acf))):
                acf_array.append(acf)
                pacf_array.append(pacf)
                rain_sorted_array.append(np.sort(y))
            
        colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        cycle = cycler.cycler(color=[colours[0]],
                              linewidth=[0.5],
                              alpha=[0.25])
        n = len(time_series)
        cdf = np.asarray(range(n))
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle)
        for y in rain_sorted_array:
            if y[-1] < self.max_rain:
                plt.plot(np.concatenate(([0], y, [self.max_rain])),
                         np.concatenate(([0], cdf, [n])))
            else:
                plt.plot(np.concatenate(([0], y)), np.concatenate(([0], cdf)))
            if np.any(y == 0):
                non_zero_index = np.nonzero(y)[0][0] - 1
                plt.scatter(0, cdf[non_zero_index], alpha=0.25)
        plt.xlim(0, self.max_rain)
        plt.xlabel("rainfall (mm)")
        plt.ylabel("cumulative frequency")
        plt.savefig(path.join(figure_directory, "prior_cdf.pdf"))
        plt.close()
        
        acf = np.asarray(acf_array)
        plt.figure()
        plt.boxplot(acf[:,1:self.n_lag+1])
        plt.xlabel("lag (day)")
        plt.ylabel("autocorrelation")
        plt.savefig(path.join(figure_directory, "prior_acf.pdf"))
        plt.close()
        
        pacf = np.asarray(pacf_array)
        plt.figure()
        plt.boxplot(pacf[:,1:self.n_lag+1])
        plt.xlabel("lag (day)")
        plt.ylabel("partial autocorrelation")
        plt.savefig(path.join(figure_directory, "prior_pacf.pdf"))
        plt.close()
    
    def __call__(self):
        self.print()

class PriorRegSimulator(PriorSimulator):
    
    def __init__(self, figure_directory, rng):
        super().__init__(figure_directory, rng)
        self.parameter_index = []
        time_series = self.get_time_series()
        parameter_name_array = time_series.get_parameter_vector_name()
        for parameter_name in parameter_name_array:
            if not(parameter_name.endswith("const")
                or "_AR" in parameter_name
                or "_MA" in parameter_name):
                self.parameter_index.append(True)
            else:
                self.parameter_index.append(False)
        self.parameter_index = np.asarray(self.parameter_index)
    
    def get_prior_time_series(self, x, prior_std):
        time_series = super().get_prior_time_series(x, prior_std)
        prior_cov_chol = time_series.parameter_target.prior_cov_chol
        prior_cov_chol[self.parameter_index] = prior_std
        return time_series
    
    def marginalise_time_series(self, time_series):
        prior_cov_chol = time_series.parameter_target.prior_cov_chol
        prior_cov_chol[np.logical_not(self.parameter_index)] = 0
    
    def __call__(self):
        std_const_array = np.linspace(0, 2, 11)
        for i, std_const in enumerate(std_const_array):
            figure_directory_i = path.join(self.figure_directory, str(i))
            self.print(figure_directory_i, std_const)
            file = open(path.join(figure_directory_i, "std.txt"), "w")
            file.write(str(std_const))
            file.close()
        self.print()

class PriorConstSimulator(PriorSimulator):
    
    def __init__(self, figure_directory, rng):
        super().__init__(figure_directory, rng)
        self.parameter_index = []
        time_series = self.get_time_series()
        parameter_name_array = time_series.get_parameter_vector_name()
        for parameter_name in parameter_name_array:
            if parameter_name.endswith("const"):
                self.parameter_index.append(True)
            else:
                self.parameter_index.append(False)
        self.parameter_index = np.asarray(self.parameter_index)
        self.prior_mean = time_series.parameter_target.prior_mean
    
    def simulate_prior_time_series(self, x, prior_std):
        time_series = compound_poisson.TimeSeriesMcmc(
            x, poisson_rate_n_arma=self.n_arma,
            gamma_mean_n_arma=self.n_arma)
        time_series.rng = self.rng
        prior_cov_chol = time_series.parameter_target.prior_cov_chol
        prior_cov_chol[np.logical_not(self.parameter_index)] = 0
        time_series.simulate_from_prior()
        return time_series
    
    def simulate_hyper_time_series(self, x):
        time_series = compound_poisson.TimeSeriesHyperSlice(
            x, poisson_rate_n_arma=self.n_arma,
            gamma_mean_n_arma=self.n_arma)
        precision = time_series.precision_target.simulate_from_prior(self.rng)
        return self.simulate_prior_time_series(x, 1 / math.sqrt(precision[0]))
    
    def __call__(self):
        std_const_array = np.linspace(0, 2, 11)
        for i, std_const in enumerate(std_const_array):
            figure_directory_i = path.join(self.figure_directory, str(i))
            self.print(figure_directory_i, std_const)
            file = open(path.join(figure_directory_i, "std.txt"), "w")
            file.write(str(std_const))
            file.close()
        self.print()

class PriorArmaSimulator(PriorSimulator):
    
    def __init__(self, figure_directory, rng):
        super().__init__(figure_directory, rng)
        self.parameter_index = []
        time_series = self.get_time_series()
        parameter_name_array = time_series.get_parameter_vector_name()
        for parameter_name in parameter_name_array:
            if "_AR" in parameter_name or "_MA" in parameter_name:
                self.parameter_index.append(True)
            else:
                self.parameter_index.append(False)
        self.parameter_index = np.asarray(self.parameter_index)
    
    def get_prior_time_series(self, x, prior_std):
        time_series = super().get_prior_time_series(x, prior_std)
        prior_cov_chol = time_series.parameter_target.prior_cov_chol
        prior_cov_chol[self.parameter_index] = prior_std
        return time_series
    
    def marginalise_time_series(self, time_series):
        prior_cov_chol = time_series.parameter_target.prior_cov_chol
        prior_cov_chol[np.logical_not(self.parameter_index)] = 0
    
    def __call__(self):
        std_const_array = np.linspace(0, 0.5, 11)
        for i, std_const in enumerate(std_const_array):
            figure_directory_i = path.join(self.figure_directory, str(i))
            self.print(figure_directory_i, std_const)
            file = open(path.join(figure_directory_i, "std.txt"), "w")
            file.write(str(std_const))
            file.close()
        self.print()
