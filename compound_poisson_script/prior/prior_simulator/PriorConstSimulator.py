import math
import os

import numpy as np

from .PriorSimulator import PriorSimulator

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
    
    def get_prior_time_series(self, x, prior_std):
        time_series = super().get_prior_time_series(x, prior_std)
        prior_cov_chol = time_series.parameter_target.prior_cov_chol
        prior_cov_chol[self.parameter_index] = prior_std
        return time_series
    
    def marginalise_time_series(self, time_series):
        prior_cov_chol = time_series.parameter_target.prior_cov_chol
        prior_cov_chol[np.logical_not(self.parameter_index)] = 0
    
    def __call__(self):
        std_const_array = np.linspace(0, 1.0, 10)
        for i, std_const in enumerate(std_const_array):
            figure_directory_i = os.path.join(self.figure_directory, str(i))
            self.print(figure_directory_i, std_const)
            file = open(os.path.join(figure_directory_i, "std.txt"), "w")
            file.write(str(std_const))
            file.close()
        self.print()
