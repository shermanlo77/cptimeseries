import compound_poisson as cp
import numpy as np
import numpy.random as random
import pdb
import math
import matplotlib.pyplot as plot
import joblib

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

time_series = cp.TimeSeriesMcmc(x, [poisson_rate, gamma_mean, gamma_dispersion])
true_parameter = time_series.get_parameter_vector()
time_series.simulate(rng)

poisson_rate_guess = math.log(n/(n- np.count_nonzero(time_series.z_array)))
gamma_mean_guess = np.mean(time_series.y_array) / poisson_rate_guess
gamma_dispersion_guess = (np.var(time_series.y_array, ddof=1)
    /poisson_rate_guess/math.pow(gamma_mean_guess,2)-1)

poisson_rate = cp.PoissonRate(n_dim)
gamma_mean = cp.GammaMean(n_dim)
gamma_dispersion = cp.GammaDispersion(n_dim)
poisson_rate["const"] = math.log(poisson_rate_guess)
gamma_mean["const"] = math.log(gamma_mean_guess)
gamma_dispersion["const"] = math.log(gamma_dispersion_guess)

time_series.set_new_parameter([poisson_rate, gamma_mean, gamma_dispersion])
time_series.fit()
joblib.dump(time_series, "mcmc_result.zlib")

pdb.set_trace()
