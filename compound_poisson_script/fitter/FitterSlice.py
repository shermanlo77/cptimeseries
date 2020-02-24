import os

import joblib
import matplotlib.pyplot as plot
import numpy as np

from .Fitter import Fitter
import compound_poisson as cp

class FitterSlice(Fitter):
    
    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        super().__init__(dataset, rng, name, result_dir, figure_dir)
        self.time_series_class = cp.TimeSeriesSlice
        
    def print_chain(self, time_series):
        super().print_chain(time_series)
        directory = os.path.join(self.figure_dir, self.name)
        
        plot.figure()
        plot.plot(np.asarray(time_series.parameter_mcmc.n_reject_array))
        plot.ylabel("Number of rejects in parameter slicing")
        plot.xlabel("Parameter sample number")
        plot.savefig(os.path.join(directory, "n_reject_parameter.pdf"))
        plot.close()

        plot.figure()
        plot.plot(np.asarray(time_series.z_mcmc.slice_width_array))
        plot.ylabel("Latent variable slice width")
        plot.xlabel("Latent variable sample number")
        plot.savefig(os.path.join(directory, "slice_width_z.pdf"))
        plot.close()
