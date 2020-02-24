import os

import joblib
import matplotlib.pyplot as plot
import numpy as np

import dataset

class Fitter:
    
    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        self.time_series_class = None
        self.dataset = dataset
        self.rng = rng
        self.name = name
        self.result_dir = result_dir
        self.figure_dir = figure_dir
    
    def __call__(self):
        
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        if not os.path.isdir(self.figure_dir):
            os.mkdir(self.figure_dir)
        if not os.path.isdir(os.path.join(self.figure_dir, self.name)):
            os.mkdir(os.path.join(self.figure_dir, self.name))
        
        result_file = os.path.join(self.result_dir, self.name + ".gz")
        if not os.path.isfile(result_file):
            model_field, rain = self.dataset.get_data_training()
            time_series = self.time_series_class(
                model_field, rain, (5, 5), (5, 5))
            time_series.time_array = self.dataset.get_time_training()
            time_series.rng = self.rng
            time_series.fit()
            joblib.dump(time_series, result_file)
        else:    
            time_series = joblib.load(result_file)
        
        self.print_chain(time_series)
        
    def print_chain(self, time_series):
        directory = os.path.join(self.figure_dir, self.name)
        parameter_name = time_series.get_parameter_vector_name()
        
        try:
            true_parameter = self.dataset.time_series.get_parameter_vector()
            is_has_true = True
        except AttributeError:
            is_has_true = False
        
        chain = np.asarray(time_series.parameter_mcmc.sample_array)
        for i in range(time_series.n_parameter):
            chain_i = chain[:,i]
            plot.figure()
            plot.plot(chain_i)
            if is_has_true:
                plot.hlines(true_parameter[i], 0, len(chain)-1)
            plot.ylabel(parameter_name[i])
            plot.xlabel("Sample number")
            plot.savefig(
                os.path.join(directory, "chain_parameter_" + str(i) + ".pdf"))
            plot.close()
        
        chain = []
        z_chain = np.asarray(time_series.z_mcmc.sample_array)
        for z in z_chain:
            chain.append(np.mean(z))
        plot.figure()
        plot.plot(chain)
        plot.ylabel("Mean of latent variables")
        plot.xlabel("Sample number")
        plot.savefig(os.path.join(directory, "chain_z.pdf"))
        plot.close()
