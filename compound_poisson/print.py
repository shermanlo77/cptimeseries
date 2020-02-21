import os

import joblib
import math
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import statsmodels.tsa.stattools as stats
from cycler import cycler
from pandas.plotting import register_matplotlib_converters

def time_series(time_series, directory, prefix=""):
    
    register_matplotlib_converters()
    
    colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    cycle = cycler(color=[colours[0]], linewidth=[1])
    
    x = time_series.x
    y = time_series.y_array
    z = time_series.z_array
    n = len(time_series)
    n_model_field = time_series.n_model_field
    t = time_series.time_array
    poisson_rate_array = time_series.poisson_rate.value_array
    gamma_mean_array = time_series.gamma_mean.value_array
    gamma_dispersion_array = time_series.gamma_dispersion.value_array
    
    acf = stats.acf(y, nlags=20, fft=True)
    try:
        pacf = stats.pacf(y, nlags=20)
    except(stats.LinAlgError):
        pacf = np.full(21, np.nan)
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.plot(t, y)
    plot.xlabel("time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(os.path.join(directory, prefix + "rainfall.pdf"))
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    rain_sorted = np.sort(y)
    cdf = np.asarray(range(n))
    plot.plot(rain_sorted, cdf)
    if np.any(rain_sorted == 0):
        non_zero_index = rain_sorted.nonzero()[0]
        if non_zero_index.size > 0:
            non_zero_index = rain_sorted.nonzero()[0][0] - 1
        else:
            non_zero_index = len(cdf) - 1
        plot.scatter(0, cdf[non_zero_index])
    plot.xlabel("rainfall (mm)")
    plot.ylabel("cumulative frequency")
    plot.savefig(os.path.join(directory, prefix + "cdf.pdf"))
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.bar(np.asarray(range(acf.size)), acf)
    plot.axhline(1/math.sqrt(n), linestyle='--', linewidth=1)
    plot.axhline(-1/math.sqrt(n), linestyle='--', linewidth=1)
    plot.xlabel("time (day)")
    plot.ylabel("autocorrelation")
    plot.savefig(os.path.join(directory, prefix + "acf.pdf"))
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.bar(np.asarray(range(pacf.size)), pacf)
    plot.axhline(1/math.sqrt(n), linestyle='--', linewidth=1)
    plot.axhline(-1/math.sqrt(n), linestyle='--', linewidth=1)
    plot.xlabel("time (day)")
    plot.ylabel("partial autocorrelation")
    plot.savefig(os.path.join(directory, prefix + "pacf.pdf"))
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.plot(t, poisson_rate_array)
    plot.xlabel("time")
    plot.ylabel("poisson rate")
    plot.savefig(os.path.join(directory, prefix + "poisson_rate.pdf"))
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.plot(t, gamma_mean_array)
    plot.xlabel("time")
    plot.ylabel("gamma mean (mm)")
    plot.savefig(os.path.join(directory, prefix + "gamma_mean.pdf"))
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.plot(t, gamma_dispersion_array)
    plot.xlabel("time")
    plot.ylabel("gamma dispersion")
    plot.savefig(os.path.join(directory, prefix + "gamma_dispersion.pdf"))
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.plot(t, z)
    plot.xlabel("time")
    plot.ylabel("Z")
    plot.savefig(os.path.join(directory, prefix + "z.pdf"))
    plot.close()
    
    file = open(os.path.join(directory, prefix + "parameter.txt"), "w")
    file.write(str(time_series))
    file.close()

def forecast(forecast, observed_rain, directory, prefix=""):
    
    register_matplotlib_converters()
    rain_threshold_array = [0, 5, 10, 15]
    
    colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    cycle_forecast = cycler(color=[colours[1], colours[0]],
                            linewidth=[1, 1],
                            alpha=[1, 0.5])
    
    time_array = forecast.time_array
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle_forecast)
    plot.fill_between(time_array,
                      forecast.forecast_sigma[-1],
                      forecast.forecast_sigma[1],
                      alpha=0.25)
    plot.plot(time_array, forecast.forecast)
    plot.plot(time_array, observed_rain)
    plot.xlabel("Time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(os.path.join(directory, prefix + "_forecast.pdf"))
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle_forecast)
    plot.fill_between(time_array,
                      forecast.forecast_sigma[-1],
                      forecast.forecast_sigma[1],
                      alpha=0.25)
    plot.plot(time_array, forecast.forecast_median)
    plot.plot(time_array, observed_rain)
    plot.xlabel("Time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(os.path.join(directory, prefix + "_forecast_median.pdf"))
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle_forecast)
    plot.plot(time_array, forecast.forecast_sigma[2])
    plot.plot(time_array, forecast.forecast_median)
    plot.xlabel("Time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(os.path.join(directory, prefix + "_forecast_extreme.pdf"))
    plot.close()
    
    plot.figure()
    plot.plot(time_array, forecast.forecast - observed_rain)
    plot.xlabel("Time")
    plot.ylabel("residual (mm)")
    plot.savefig(os.path.join(directory, prefix + "_residual.pdf"))
    plot.close()
    
    plot.figure()
    for rain in rain_threshold_array:
        forecast.plot_roc_curve(rain, observed_rain)
    plot.legend()
    plot.savefig(os.path.join(directory, prefix + "_roc.pdf"))
    plot.close()
    
    for rain in rain_threshold_array:
        plot.figure()
        plot.plot(time_array, forecast.get_prob_rain(rain))
        for day in range(forecast.n):
            if observed_rain[day] > rain:
                plot.axvline(x=time_array[day], color="r", linestyle=":")
        plot.xlabel("time")
        plot.ylabel("forecasted probability of > "+str(rain)+" mm of rain")
        plot.savefig(
            os.path.join(directory, prefix + "_prob_" + str(rain) + ".pdf"))
        plot.close()

    file = open(os.path.join(directory, prefix + "_errors.txt"), "w")
    file.write("Deviance: ")
    file.write(str(forecast.get_error_square_sqrt(observed_rain)))
    file.write("\n")
    file.write("Rmse: ")
    file.write(str(forecast.get_error_rmse(observed_rain)))
    file.write("\n")
    file.close()
