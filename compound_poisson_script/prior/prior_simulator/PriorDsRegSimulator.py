import math
import os

import numpy as np

from .PriorDsSimulator import PriorDsSimulator

class PriorDsRegSimulator(PriorDsSimulator):
    
    def __init__(self, figure_directory, rng):
        super().__init__(figure_directory, rng)
    
    def simulate(self, precision):
        downscale = self.downscale
        downscale.gp_target.precision = precision
        downscale.precision_target.precision = (
            downscale.precision_target.simulate_from_prior(self.rng))
        downscale.update_reg_gp()
        downscale.update_reg_precision()
        parameter = downscale.parameter_target.simulate_from_prior(self.rng)
        downscale.set_parameter_vector(parameter)
        downscale.simulate_i(0)
    
    def __call__(self):
        precision_array = np.linspace(2.27, 20, 10)
        for i, precision in enumerate(precision_array):
            figure_directory_i = os.path.join(self.figure_directory, str(i))
            self.print(figure_directory_i, precision)
            file = open(os.path.join(figure_directory_i, "precision.txt"), "w")
            file.write(str(precision))
            file.close()
