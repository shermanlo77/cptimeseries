import compound_poisson as cp
import math
import numpy as np
import numpy.random as random

def simulate_training(TimeSeriesClass=None):
    
    if TimeSeriesClass is None:
        TimeSeriesClass = cp.TimeSeries
    
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
    
    time_series = TimeSeriesClass(
        x, [poisson_rate, gamma_mean, gamma_dispersion])
    time_series.rng = rng
    time_series.simulate()
    
    return time_series

def simulate_test():
    time_series = simulate_training(cp.TimeSeries)
    time_series.rng = random.RandomState(np.uint32(162715606))
    rng = time_series.rng
    n = len(time_series)
    n_dim = time_series.n_dim
    n_forecast = 10000
    x = np.zeros((n_forecast,n_dim))
    for i in range(n_forecast):
        for i_dim in range(n_dim):
            x[i, i_dim] = (0.8*math.sin(2*math.pi/365 * (i+n))
                + (i_dim+1)*rng.normal())
    return time_series.simulate_future(x)
