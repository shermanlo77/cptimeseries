import compound_poisson as cp
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import statsmodels.tsa.stattools as stats

def main():
    
    rng = random.RandomState(np.uint32(372625178))
    n = 10000
    n_dim = 2
    x = np.zeros((n,n_dim))
    for i in range(n):
        for i_dim in range(n_dim):
            x[i, i_dim] = (0.8*math.sin(2*math.pi/365 * i)
                + (i_dim+1)*rng.normal())
    
    poisson_rate = cp.PoissonRate(n_dim)
    poisson_rate["reg"] = [0.098, 0.001]
    poisson_rate["AR"] = 0.13
    poisson_rate["MA"] = 0.19
    poisson_rate["const"] = 0.42
    gamma_mean = cp.GammaMean(n_dim)
    gamma_mean["reg"] = [0.066, 0.002]
    gamma_mean["AR"] = 0.1
    gamma_mean["MA"] = 0.1
    gamma_mean["const"] = 0.89
    gamma_dispersion = cp.GammaDispersion(n_dim)
    gamma_dispersion["reg"] = [0.07, 0.007]
    gamma_dispersion["const"] = 0.12
    
    time_series = cp.TimeSeriesSgd(x, [poisson_rate, gamma_mean, gamma_dispersion])
    time_series.simulate(rng)
    print_figures(time_series, "simulation")

def print_figures(time_series, prefix):
    
    x = time_series.x
    y = time_series.y_array
    z = time_series.z_array
    n = time_series.n
    n_dim = time_series.n_dim
    poisson_rate_array = time_series.poisson_rate.value_array
    gamma_mean_array = time_series.gamma_mean.value_array
    gamma_dispersion_array = time_series.gamma_dispersion.value_array
    
    acf = stats.acf(y, nlags=100, fft=True)
    pacf = stats.pacf(y, nlags=10)
    
    plt.figure()
    plt.plot(y)
    plt.xlabel("Time (day)")
    plt.ylabel("Rainfall (mm)")
    plt.show()
    plt.close()
    
    for i_dim in range(n_dim):
        plt.figure()
        plt.plot(x[:,i_dim])
        plt.xlabel("Time (day)")
        plt.ylabel("Model field "+str(i_dim))
        plt.show()
        plt.close()
    
    plt.figure()
    plt.bar(np.asarray(range(acf.size)), acf)
    plt.xlabel("Time (day)")
    plt.ylabel("Autocorrelation")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.bar(np.asarray(range(pacf.size)), pacf)
    plt.xlabel("Time (day)")
    plt.ylabel("Partial autocorrelation")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(poisson_rate_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Poisson rate")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(gamma_mean_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Gamma mean (mm)")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(gamma_dispersion_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Gamma dispersion")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(range(n), z)
    plt.xlabel("Time (day)")
    plt.ylabel("Z")
    plt.show()
    plt.close()

main()
