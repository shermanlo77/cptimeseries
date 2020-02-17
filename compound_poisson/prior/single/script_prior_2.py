import os
from get_data import RandomLocation, London
import compound_poisson as cp
import math
import numpy as np
import numpy.random as random
from scipy.stats import gamma

import matplotlib
import matplotlib.pyplot as plot
import statsmodels.tsa.stattools as stats
from cycler import cycler
from print_figure import print_time_series

def main():
    
    figure_directory = os.path.join("prior_figures", "final_arma")
    try:
        os.mkdir(figure_directory)
    except(FileExistsError):
        pass
    
    rng = random.RandomState(np.uint32(931388699))
    precision_arma = gamma(a=1.3, loc=16, scale=65)
    precision_arma.random_state = rng
    
    n_arma = (10, 10)
    n_simulation = 100
    n_lag = 20
    max_rain = 100
    
    london = London()
    model_field = london.get_model_field_training()
    time_series = cp.TimeSeriesMcmc(model_field, None, n_arma, n_arma)
    time_series.rng = rng
    parameter_name_array = time_series.get_parameter_vector_name()
    
    acf_array = []
    pacf_array = []
    rain_sorted_array = []
    
    for i_simulate in range(n_simulation):
        
        var_arma = 1/precision_arma.rvs()
        var_reg = 0
        
        time_series.prior_covariance = np.zeros(time_series.n_parameter)
        for dim, parameter_name in enumerate(parameter_name_array):
            if parameter_name.endswith("const"):
                time_series.prior_covariance[dim] = var_reg
            elif "_AR" in parameter_name:
                time_series.prior_covariance[dim] = var_arma
            elif "_MA" in parameter_name:
                time_series.prior_covariance[dim] = var_arma
            else:
                time_series.prior_covariance[dim] = var_reg
        
        time_series.simulate_from_prior()
        print_time_series(
            time_series,
            os.path.join(figure_directory, str(i_simulate) + "_"))
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
    n = len(time_series)
    cdf = np.asarray(range(n))
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
    plot.savefig(os.path.join(figure_directory, "prior_cdf.pdf"))
    plot.close()
    
    acf = np.asarray(acf_array)
    plot.figure()
    plot.boxplot(acf[:,1:n_lag+1])
    plot.xlabel("lag (day)")
    plot.ylabel("autocorrelation")
    plot.savefig(os.path.join(figure_directory, "prior_acf.pdf"))
    plot.close()
    
    pacf = np.asarray(pacf_array)
    plot.figure()
    plot.boxplot(pacf[:,1:n_lag+1])
    plot.xlabel("lag (day)")
    plot.ylabel("partial autocorrelation")
    plot.savefig(os.path.join(figure_directory, "prior_pacf.pdf"))
    plot.close()

if __name__ == "__main__":
    main()
 
