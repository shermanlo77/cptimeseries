import math
import os
import sys

import numpy as np

from .PriorSimulator import PriorSimulator
import compound_poisson as cp

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
        time_series = cp.TimeSeriesMcmc(
            x, poisson_rate_n_arma=self.n_arma,
            gamma_mean_n_arma=self.n_arma)
        time_series.rng = self.rng
        prior_cov_chol = time_series.parameter_target.prior_cov_chol
        prior_cov_chol[np.logical_not(self.parameter_index)] = 0
        time_series.simulate_from_prior()
        return time_series
    
    def simulate_hyper_time_series(self, x):
        time_series = cp.TimeSeriesHyperSlice(
            x, poisson_rate_n_arma=self.n_arma,
            gamma_mean_n_arma=self.n_arma)
        precision = time_series.precision_target.simulate_from_prior(self.rng)
        return self.simulate_prior_time_series(x, 1 / math.sqrt(precision[0]))
    
    def __call__(self):
        std_const_array = np.linspace(0, 2, 11)
        for i, std_const in enumerate(std_const_array):
            figure_directory_i = os.path.join(self.figure_directory, str(i))
            self.print(figure_directory_i, std_const)
            file = open(os.path.join(figure_directory_i, "std.txt"), "w")
            file.write(str(std_const))
            file.close()
        self.print()
