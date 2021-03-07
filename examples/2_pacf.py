"""Script showing an example of simulated compound-Poisson with AR and MA terms
    and no no model fields. Illustrate ARMA nature of this model with a plot of
    the time series, the autocorrelation and partial autocorrelation.

It should be noted that the implementation of ARMA in this model is an
    emulation, ie it's not really ARMA but there was an attempt to mimic the
    behaviour. The acf and pacf can be used to look for peaks at certain lags,
    a characteristic of ARMA. For this specific emulation, having only AR terms
    is not possible.

    For other emulations, see for example:
        Benjamin, M.A. and Stasinopoulos, M., 1998. Modelling exponential family
        time series data. In Statistical Modelling: Proceedings of the 13th
        International Workshop on Stastical Modelling.
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
    n_arma = [1, 1] #sets number of ar and ma terms to be 0 and 1
    #value of the ma parameter
    ar_parameter = np.asarray([0.5])

    #set seed of the rng
    seed = random.SeedSequence(149516089283625725195385184466521592500)
    rng = random.RandomState(random.MT19937(seed))

    #define the parameters for this model
    poisson_rate = parameter.PoissonRate(n_model_field, n_arma)
    gamma_mean = parameter.GammaMean(n_model_field, n_arma)
    gamma_dispersion = parameter.GammaDispersion(n_model_field)

    #set the ma parameter
    poisson_rate["AR"] = ar_parameter
    gamma_mean["AR"] = ar_parameter
    poisson_rate["MA"] = ar_parameter
    gamma_mean["MA"] = ar_parameter

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
    plt.title("Compound-Poisson with ARMA(1, 1)")
    plt.xlabel("time (days)")
    plt.ylabel("precipitation (mm)")
    plt.show()
    plt.close()

    #plt the sample autocorrelation
    #a peak at lag 1 indicate MA(1) behaviour
    acf = stattools.acf(time_series[:])
    plt.figure()
    plt.bar(range(len(acf)), acf)
    plt.title("Compound-Poisson with ARMA(1, 1)")
    plt.xlabel("lag (days)")
    plt.ylabel("autocorrelation")
    plt.show()
    plt.close()

    #plt the sample partial autocorrelation
    #a peak at lag 1 indicate AR(1) behaviour
    pacf = stattools.pacf_yw(time_series[:])
    plt.figure()
    plt.bar(range(len(pacf)), pacf)
    plt.title("Compound-Poisson with ARMA(1, 1)")
    plt.xlabel("lag (days)")
    plt.ylabel("partial autocorrelation")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
