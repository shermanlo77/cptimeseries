import os

import joblib
import matplotlib.pyplot as plot
import numpy as np

from .Fitter import Fitter
import compound_poisson as cp

class FitterMcmc(Fitter):
    
    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        super().__init__(dataset, rng, name, result_dir, figure_dir)
        self.time_series_class = cp.TimeSeriesMcmc
        
    def print_chain(self, time_series):
        super().print_chain(time_series)
        directory = os.path.join(self.figure_dir, self.name)
        
        plot.figure()
        plot.plot(np.asarray(time_series.parameter_mcmc.accept_array))
        plot.ylabel("Acceptance rate of parameters")
        plot.xlabel("Parameter sample number")
        plot.savefig(os.path.join(directory, "accept_parameter.pdf"))
        plot.close()

        plot.figure()
        plot.plot(np.asarray(time_series.z_mcmc.accept_array))
        plot.ylabel("Acceptance rate of latent variables")
        plot.xlabel("Latent variable sample number")
        plot.savefig(os.path.join(directory, "accept_z.pdf"))
        plot.close()
