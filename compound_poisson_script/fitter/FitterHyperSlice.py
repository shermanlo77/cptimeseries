import os

import joblib
import matplotlib.pyplot as plot
import numpy as np

from .FitterSlice import FitterSlice
import compound_poisson as cp

class FitterHyperSlice(FitterSlice):
    
    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        super().__init__(dataset, rng, name, result_dir, figure_dir)
        self.time_series_class = cp.TimeSeriesHyperSlice
        
    def print_chain(self, time_series):
        super().print_chain(time_series)
        directory = os.path.join(self.figure_dir, self.name)
        
        precision_chain = np.asarray(time_series.precision_mcmc.sample_array)
        for i in range(2):
            chain_i = precision_chain[:, i]
            plot.figure()
            plot.plot(chain_i)
            plot.ylabel("precision" + str(i))
            plot.xlabel("Sample number")
            plot.savefig(
                os.path.join(directory, "chain_precision_" + str(i) + ".pdf"))
            plot.close()
            
        plot.figure()
        plot.plot(np.asarray(time_series.precision_mcmc.accept_array))
        plot.ylabel("Acceptance rate of parameters")
        plot.xlabel("Parameter sample number")
        plot.savefig(os.path.join(directory, "accept_precision.pdf"))
        plot.close()
