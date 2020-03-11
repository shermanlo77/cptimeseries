import os
from os import path

import matplotlib.pyplot as plt

import compound_poisson

class Fitter:
    
    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        self.time_series_class = None
        self.dataset = dataset
        self.rng = rng
        self.name = name
        self.result_dir = result_dir
        self.figure_dir = figure_dir
    
    def __call__(self):
        
        figure_sub_dir = path.join(self.figure_dir, self.name)
        if not path.isdir(figure_sub_dir):
            os.mkdir(figure_sub_dir)
        
        result_file = path.join(self.result_dir, self.name + ".gz")
        if not path.isfile(result_file):
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
        directory = path.join(self.figure_dir, self.name)
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
                path.join(directory, "chain_parameter_" + str(i) + ".pdf"))
            plot.close()
        
        chain = []
        z_chain = np.asarray(time_series.z_mcmc.sample_array)
        for z in z_chain:
            chain.append(np.mean(z))
        plot.figure()
        plot.plot(chain)
        plot.ylabel("Mean of latent variables")
        plot.xlabel("Sample number")
        plot.savefig(path.join(directory, "chain_z.pdf"))
        plot.close()

class FitterMcmc(Fitter):
    
    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        super().__init__(dataset, rng, name, result_dir, figure_dir)
        self.time_series_class = compound_poisson.TimeSeriesMcmc
        
    def print_chain(self, time_series):
        super().print_chain(time_series)
        directory = path.join(self.figure_dir, self.name)
        
        plt.figure()
        plt.plot(np.asarray(time_series.parameter_mcmc.accept_array))
        plt.ylabel("Acceptance rate of parameters")
        plt.xlabel("Parameter sample number")
        plt.savefig(path.join(directory, "accept_parameter.pdf"))
        plt.close()

        plt.figure()
        plt.plot(np.asarray(time_series.z_mcmc.accept_array))
        plt.ylabel("Acceptance rate of latent variables")
        plt.xlabel("Latent variable sample number")
        plt.savefig(path.join(directory, "accept_z.pdf"))
        plt.close()

class FitterSlice(Fitter):
    
    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        super().__init__(dataset, rng, name, result_dir, figure_dir)
        self.time_series_class = compound_poisson.TimeSeriesSlice
        
    def print_chain(self, time_series):
        super().print_chain(time_series)
        directory = path.join(self.figure_dir, self.name)
        
        plot.figure()
        plot.plot(np.asarray(time_series.parameter_mcmc.n_reject_array))
        plot.ylabel("Number of rejects in parameter slicing")
        plot.xlabel("Parameter sample number")
        plot.savefig(path.join(directory, "n_reject_parameter.pdf"))
        plot.close()

        plot.figure()
        plot.plot(np.asarray(time_series.z_mcmc.slice_width_array))
        plot.ylabel("Latent variable slice width")
        plot.xlabel("Latent variable sample number")
        plot.savefig(path.join(directory, "slice_width_z.pdf"))
        plot.close()

class FitterHyperSlice(FitterSlice):
    
    def __init__(self, dataset, rng, name, result_dir, figure_dir):
        super().__init__(dataset, rng, name, result_dir, figure_dir)
        self.time_series_class = compound_poisson.TimeSeriesHyperSlice
        
    def print_chain(self, time_series):
        super().print_chain(time_series)
        directory = path.join(self.figure_dir, self.name)
        
        precision_chain = np.asarray(time_series.precision_mcmc.sample_array)
        for i in range(2):
            chain_i = precision_chain[:, i]
            plt.figure()
            plt.plot(chain_i)
            plt.ylabel("precision" + str(i))
            plt.xlabel("Sample number")
            plt.savefig(
                path.join(directory, "chain_precision_" + str(i) + ".pdf"))
            plt.close()
            
        plt.figure()
        plt.plot(np.asarray(time_series.precision_mcmc.accept_array))
        plt.ylabel("Acceptance rate of parameters")
        plt.xlabel("Parameter sample number")
        plt.savefig(path.join(directory, "accept_precision.pdf"))
        plt.close()
