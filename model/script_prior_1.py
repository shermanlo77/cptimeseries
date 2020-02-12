import os
from get_data import RandomLocation, London
import compound_poisson as cp
import math
import numpy as np
import numpy.random as random

import matplotlib
import matplotlib.pyplot as plot
import statsmodels.tsa.stattools as stats
from cycler import cycler
from print_figure import print_time_series

def main():
    
    figure_directory = "prior_figures"
    const_directory = os.path.join(figure_directory, "const")
    arma_directory = os.path.join(figure_directory, "arma")
    model_field_directory = os.path.join(figure_directory, "model_field")
    rng = random.RandomState(np.uint32(4153451458))
    
    try:
        os.mkdir(figure_directory)
    except(FileExistsError):
        pass
    try:
        os.mkdir(const_directory)
    except(FileExistsError):
        pass
    try:
        os.mkdir(arma_directory)
    except(FileExistsError):
        pass
    try:
        os.mkdir(model_field_directory)
    except(FileExistsError):
        pass
    
    n_arma = (10, 10)
    n_simulation = 100
    n_lag = 20
    max_rain = 100
    
    std_const = np.linspace(0, 2, 10)
    std_arma = np.linspace(0, 0.5, 10)
    std_model_field = np.linspace(0, 1.0, 10)
    
    directory_array = [const_directory, arma_directory, model_field_directory]
    std_array = [std_const, std_arma, std_model_field]
    
    london = London()
    model_field = london.get_model_field_training()
    time_series = cp.TimeSeriesMcmc(model_field, None, n_arma, n_arma)
    time_series.rng = rng
    n = len(model_field)
    cdf = np.asarray(range(n))
    
    random_location = RandomLocation(rng)
    
    for i_parameter in range(len(directory_array)):
    
        parameter_name_array = time_series.get_parameter_vector_name()
        parameter_index = []
        
        for dim, parameter_name in enumerate(parameter_name_array):
            if parameter_name.endswith("const"):
                if i_parameter == 0:
                    parameter_index.append(dim)
            elif "_AR" in parameter_name:
                if i_parameter == 1:
                    parameter_index.append(dim)
            elif "_MA" in parameter_name:
                if i_parameter == 1:
                    parameter_index.append(dim)
            else:
                if i_parameter == 2:
                    parameter_index.append(dim)
        
        for i_std, std in enumerate(std_array[i_parameter]):
            
            i_directory = os.path.join(directory_array[i_parameter], str(i_std))
            try:
                os.mkdir(i_directory)
            except(FileExistsError):
                pass
            
            time_series.prior_covariance = np.zeros(time_series.n_parameter)
            for dim in parameter_index:
                time_series.prior_covariance[dim] = math.pow(std, 2)
            
            file = open(os.path.join(i_directory, "std.txt"), "w")
            file.write(str(std))
            file.close()
            
            acf_array = []
            pacf_array = []
            rain_sorted_array = []
            
            for i_simulate in range(n_simulation):
                
                if i_parameter == 2:
                    
                    random_location.set_new_location()
                    print(random_location.coordinates)
                    model_field = random_location.get_model_field_training()
                    time_series = cp.TimeSeriesMcmc(
                        model_field, None, n_arma, n_arma)
                    time_series.rng = rng
                    time_series.prior_covariance = np.zeros(
                        time_series.n_parameter)
                    for dim in parameter_index:
                        time_series.prior_covariance[dim] = math.pow(std, 2)
                
                time_series.simulate_from_prior()
                print_time_series(
                    time_series,
                    os.path.join(i_directory, str(i_simulate) + "_"))
                y = time_series.y_array
                acf = stats.acf(y, nlags=n_lag, fft=True)
                try:
                    pacf = stats.pacf(y, nlags=n_lag)
                except(stats.LinAlgError):
                    pacf = np.full(n_lag+1, np.nan)
                if (len(acf) == n_lag + 1) and (not np.any(np.isnan(acf))):
                    acf_array.append(acf)
                    pacf_array.append(pacf)
                    rain_sorted_array.append(np.sort(y))
            
            colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
            cycle = cycler(color=[colours[0]],
                                    linewidth=[0.5],
                                    alpha=[0.25])
            plot.figure()
            ax = plot.gca()
            ax.set_prop_cycle(cycle)
            for y in rain_sorted_array:
                
                if y[-1] < max_rain:
                    plot.plot(np.concatenate(([0], y, [max_rain])),
                              np.concatenate(([0], cdf, [n])))
                else:
                    plot.plot(np.concatenate(([0], y)),
                              np.concatenate(([0], cdf)))
                if np.any(y == 0):
                    non_zero_index = np.nonzero(y)[0][0] - 1
                    plot.scatter(0, cdf[non_zero_index], alpha=0.25)
            plot.xlim(0, max_rain)
            plot.xlabel("rainfall (mm)")
            plot.ylabel("cumulative frequency")
            plot.savefig(os.path.join(i_directory, "prior_cdf.pdf"))
            plot.close()
            
            acf = np.asarray(acf_array)
            plot.figure()
            plot.boxplot(acf[:,1:n_lag+1])
            plot.xlabel("lag (day)")
            plot.ylabel("autocorrelation")
            plot.savefig(os.path.join(i_directory, "prior_acf.pdf"))
            plot.close()
            
            pacf = np.asarray(pacf_array)
            plot.figure()
            plot.boxplot(pacf[:,1:n_lag+1])
            plot.xlabel("lag (day)")
            plot.ylabel("partial autocorrelation")
            plot.savefig(os.path.join(i_directory, "prior_pacf.pdf"))
            plot.close()
    
    

if __name__ == "__main__":
    main()
 
