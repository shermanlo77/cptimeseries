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
    print(time_series.poisson_rate)
    print(time_series.gamma_mean)
    print(time_series.gamma_dispersion)
    time_series.fit()
    
    plt.figure()
    ax = plt.gca()
    ln_l_array = time_series.ln_l_array
    ln_l_stochastic_index = time_series.ln_l_stochastic_index
    for i in range(len(ln_l_stochastic_index)-1):
        start = ln_l_stochastic_index[i]-1
        end = ln_l_stochastic_index[i+1]
        if i%2 == 0:
            linestyle = "-"
        else:
            linestyle = ":"
        ax.set_prop_cycle(None)
        plt.plot(range(start, end), ln_l_array[start:end], linestyle=linestyle)
    plt.axvline(x=time_series.ln_l_max_index, linestyle='--')
    plt.xlabel("Number of EM steps")
    plt.ylabel("log-likelihood")
    plt.savefig("../figures/fit_ln_l.pdf")
    plt.close()
    print(time_series.poisson_rate)
    print(time_series.gamma_mean)
    print(time_series.gamma_dispersion)
    
    time_series.simulate(rng)
    print_figures(time_series, "fitted")

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
    plt.savefig("../figures/"+prefix+"_rain.pdf")
    plt.close()
    
    for i_dim in range(n_dim):
        plt.figure()
        plt.plot(x[:,i_dim])
        plt.xlabel("Time (day)")
        plt.ylabel("Model field "+str(i_dim))
        plt.savefig("../figures/"+prefix+"_model_field_"+str(i_dim)+".pdf")
        plt.close()
    
    plt.figure()
    plt.bar(np.asarray(range(acf.size)), acf)
    plt.xlabel("Time (day)")
    plt.ylabel("Autocorrelation")
    plt.savefig("../figures/"+prefix+"_acf.pdf")
    plt.close()
    
    plt.figure()
    plt.bar(np.asarray(range(pacf.size)), pacf)
    plt.xlabel("Time (day)")
    plt.ylabel("Partial autocorrelation")
    plt.savefig("../figures/"+prefix+"_pacf.pdf")
    plt.close()
    
    plt.figure()
    plt.plot(poisson_rate_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Poisson rate")
    plt.savefig("../figures/"+prefix+"_lambda.pdf")
    plt.close()
    
    plt.figure()
    plt.plot(gamma_mean_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Gamma mean (mm)")
    plt.savefig("../figures/"+prefix+"_mu.pdf")
    plt.close()
    
    plt.figure()
    plt.plot(gamma_dispersion_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Gamma dispersion")
    plt.savefig("../figures/"+prefix+"_dispersion.pdf")
    plt.close()
    
    plt.figure()
    plt.plot(range(n), z)
    plt.xlabel("Time (day)")
    plt.ylabel("Z")
    plt.savefig("../figures/"+prefix+"_z.pdf")
    plt.close()

main()
