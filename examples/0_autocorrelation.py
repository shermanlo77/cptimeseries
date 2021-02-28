"""Script showing an example of simulated compound-Poisson with MA terms and no
    no model fields. Illustrate ARMA nature of this model with a plot of the
    time series and the autocorrelation.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from statsmodels.tsa import stattools

import compound_poisson
from compound_poisson import parameter

def main():

    time_length = 365 #length of the time series
    #no model fields, set it to one model field, filled with zeros
    n_model_field = 1
    x_array = np.zeros((time_length, n_model_field))
    n_arma = [0, 1] #sets number of ar and ma terms to be 0 and 1
    #value of the ma parameter
    ma_parameter = np.asarray([0.3])

    #set seed of the rng
    seed = random.SeedSequence(103616317136878112071633291725501775781)
    rng = random.RandomState(random.MT19937(seed))

    #define the parameters for this model
    poisson_rate = parameter.PoissonRate(n_model_field, n_arma)
    gamma_mean = parameter.GammaMean(n_model_field, n_arma)
    gamma_dispersion = parameter.GammaDispersion(n_model_field)

    #set the ma parameter
    poisson_rate["MA"] = ma_parameter
    gamma_mean["MA"] = ma_parameter

    #instantiate the time series
    parameter_array = [
        poisson_rate,
        gamma_mean,
        gamma_dispersion,
    ]
    time_series = compound_poisson.TimeSeries(
        x_array, cp_parameter_array=parameter_array)
    #set the x_shift and x_scale as by default, TimeSeries normalise the model
        #fields using mean and std. Since std of all zeros is 0, set x_scale
        #to an appropriate value
    time_series.x_shift = 0
    time_series.x_scale = 1
    time_series.rng = rng #set rng
    time_series.simulate() #and simulate

    #plot the time series
    plt.figure()
    plt.plot(time_series[:])
    plt.title("Compound-Poisson with MA(1)")
    plt.xlabel("time (days)")
    plt.ylabel("precipitation (mm)")
    plt.show()
    plt.close()

    #plt the sample autocorrelation
    #a peak at lag 1 indicate MA(1) behaviour
    acf = stattools.acf(time_series[:])
    plt.figure()
    plt.bar(range(len(acf)), acf)
    plt.title("Compound-Poisson with MA(1)")
    plt.xlabel("lag (days)")
    plt.ylabel("autocorrelation")
    plt.show()

if __name__ == "__main__":
    main()
