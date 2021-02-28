"""Script showing an example of simulated compound-Poisson with seasonal model
    fields and no ARMA terms. Illustrate seasonal nature of this model with a
    plot of the time series and the autocorrelation.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
from statsmodels.tsa import stattools

import compound_poisson
from compound_poisson import parameter

def main():

    time_length = 2*365 #length of the time series
    #one model field with sine wave
    n_model_field = 1
    x_array = np.zeros((time_length, n_model_field))
    x_array[:, 0] = range(time_length)
    x_array = np.sin(2*math.pi*x_array/365)

    n_arma = [0, 0] #no arma
    #value of the regression parameter
    reg_parameter = np.asarray([0.8])

    #set seed of the rng
    seed = random.SeedSequence(199412950541405529670631357604770615867)
    rng = random.RandomState(random.MT19937(seed))

    #define the parameters for this model
    poisson_rate = parameter.PoissonRate(n_model_field, n_arma)
    gamma_mean = parameter.GammaMean(n_model_field, n_arma)
    gamma_dispersion = parameter.GammaDispersion(n_model_field)

    #set the ma parameter
    poisson_rate["reg"] = reg_parameter
    gamma_mean["reg"] = reg_parameter

    #instantiate the time series
    parameter_array = [
        poisson_rate,
        gamma_mean,
        gamma_dispersion,
    ]
    time_series = compound_poisson.TimeSeries(
        x_array, cp_parameter_array=parameter_array)
    time_series.rng = rng #set rng
    time_series.simulate() #and simulate

    #plot the time series
    #note the sine behaviour
    plt.figure()
    plt.plot(time_series[:])
    plt.title("Seasonal Compound-Poisson")
    plt.xlabel("time (days)")
    plt.ylabel("precipitation (mm)")
    plt.show()
    plt.close()

    #plt the sample autocorrelation
    acf = stattools.acf(time_series[:])
    plt.figure()
    plt.bar(range(len(acf)), acf)
    plt.title("Seasonal Compound-Poisson")
    plt.xlabel("lag (days)")
    plt.ylabel("autocorrelation")
    plt.show()

if __name__ == "__main__":
    main()
