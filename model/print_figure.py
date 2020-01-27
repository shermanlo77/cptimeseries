import joblib
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import statsmodels.tsa.stattools as stats
from cycler import cycler
from pandas.plotting import register_matplotlib_converters

def print_time_series(time_series, prefix):
    
    register_matplotlib_converters()
    
    cycle = cycler(linewidth=[1])
    
    x = time_series.x
    y = time_series.y_array
    z = time_series.z_array
    n = len(time_series)
    n_model_fields = time_series.n_model_fields
    t = time_series.time_array
    poisson_rate_array = time_series.poisson_rate.value_array
    gamma_mean_array = time_series.gamma_mean.value_array
    gamma_dispersion_array = time_series.gamma_dispersion.value_array
    
    acf = stats.acf(y, nlags=10, fft=True)
    pacf = stats.pacf(y, nlags=10)
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.plot(t, y)
    plot.xlabel("Time")
    plot.ylabel("Rainfall (mm)")
    plot.savefig(prefix + "rainfall.pdf")
    plot.close()
    
    for i_dim in range(n_model_fields):
        plot.figure()
        ax = plot.gca()
        ax.set_prop_cycle(cycle)
        plot.plot(t, x[:,i_dim])
        plot.xlabel("Time")
        plot.ylabel("Model field "+str(i_dim))
        plot.savefig(prefix + "model_field"+str(i_dim)+".pdf")
        plot.close()
    
    plot.figure()
    plot.bar(np.asarray(range(acf.size)), acf)
    plot.xlabel("Time")
    plot.ylabel("Autocorrelation")
    plot.savefig(prefix + "acf.pdf")
    plot.close()
    
    plot.figure()
    plot.bar(np.asarray(range(pacf.size)), pacf)
    plot.xlabel("Time")
    plot.ylabel("Partial autocorrelation")
    plot.savefig(prefix + "pacf.pdf")
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.plot(t, poisson_rate_array)
    plot.xlabel("Time")
    plot.ylabel("Poisson rate")
    plot.savefig(prefix + "poisson_rate.pdf")
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.plot(t, gamma_mean_array)
    plot.xlabel("Time")
    plot.ylabel("Gamma mean (mm)")
    plot.savefig(prefix + "gamma_mean.pdf")
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.plot(t, gamma_dispersion_array)
    plot.xlabel("Time")
    plot.ylabel("Gamma dispersion")
    plot.savefig(prefix + "gamma_dispersion.pdf")
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    plot.plot(t, z)
    plot.xlabel("Time")
    plot.ylabel("Z")
    plot.savefig(prefix + "z.pdf")
    plot.close()

def print_forecast(time_series, true_y_self, x_future, true_y_future, prefix):
    
    register_matplotlib_converters()
    
    #forecast self
    try:
        forecast_self = joblib.load(prefix + "self_forecast.zlib")
    except FileNotFoundError:
        forecast_self = time_series.forecast_self(1000)
        joblib.dump(forecast_self, prefix + "self_forecast.zlib")
        
    time_array = forecast_self.time_array
    
    colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    cycle_forecast = cycler(color=[colours[1], colours[0]],
                            linewidth=[1, 1],
                            alpha=[1, 0.5])
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle_forecast)
    plot.fill_between(time_array,
                      forecast_self.forecast_sigma[-1],
                      forecast_self.forecast_sigma[1],
                      alpha=0.25)
    plot.plot(time_array, forecast_self.forecast)
    plot.plot(time_array, true_y_self)
    plot.xlabel("Time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(prefix + "forecast_self.pdf")
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle_forecast)
    plot.fill_between(time_array,
                      forecast_self.forecast_sigma[-1],
                      forecast_self.forecast_sigma[1],
                      alpha=0.25)
    plot.plot(time_array, forecast_self.forecast_median)
    plot.plot(time_array, true_y_self)
    plot.xlabel("Time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(prefix + "forecast_self_median.pdf")
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle_forecast)
    plot.plot(time_array, forecast_self.forecast_sigma[3])
    plot.plot(time_array, true_y_self)
    plot.xlabel("Time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(prefix + "forecast_self_extreme.pdf")
    plot.close()
    
    plot.figure()
    plot.plot(time_array, forecast_self.forecast - true_y_self)
    plot.xlabel("Time")
    plot.ylabel("residual (mm)")
    plot.savefig(prefix + "residual_self.pdf")
    plot.close()
    
    #forecast
    try:
        forecast = joblib.load(prefix + "forecast.zlib")
    except FileNotFoundError:
        forecast = time_series.forecast(x_future, 1000)
        joblib.dump(forecast, prefix + "forecast.zlib")
    time_array_future = forecast.time_array
    time_array_full = []
    for i in range(len(time_array)):
        time_array_full.append(time_array[i])
    for i in range(len(time_array_future)):
        time_array_full.append(time_array_future[i])
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle_forecast)
    plot.fill_between(time_array_future,
                      forecast.forecast_sigma[-1],
                      forecast.forecast_sigma[1],
                      alpha=0.25)
    plot.plot(time_array_future, forecast.forecast)
    plot.plot(time_array_future, true_y_future)
    plot.xlabel("Time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(prefix + "forecast.pdf")
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle_forecast)
    plot.fill_between(time_array_future,
                      forecast.forecast_sigma[-1],
                      forecast.forecast_sigma[1],
                      alpha=0.25)
    plot.plot(time_array_future, forecast.forecast_median)
    plot.plot(time_array_future, true_y_future)
    plot.xlabel("Time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(prefix + "forecast_median.pdf")
    plot.close()
    
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle_forecast)
    plot.plot(time_array_future, forecast.forecast_sigma[3])
    plot.plot(time_array_future, true_y_future)
    plot.xlabel("Time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(prefix + "forecast_extreme.pdf")
    plot.close()
    
    plot.figure()
    plot.plot(time_array_future, forecast.forecast - true_y_future)
    plot.xlabel("Time")
    plot.ylabel("rainfall (mm)")
    plot.savefig(prefix + "residual.pdf")
    plot.close()
    
    file = open(prefix + "errors.txt", "w")
    file.write("Self deviance: ")
    file.write(str(forecast_self.get_error_square_sqrt(true_y_self)))
    file.write("\n")
    file.write("Self rmse: ")
    file.write(str(forecast_self.get_error_rmse(true_y_self)))
    file.write("\n")
    file.write("Forecast deviance: ")
    file.write(str(forecast.get_error_square_sqrt(true_y_future)))
    file.write("\n")
    file.write("Forecast rmse: ")
    file.write(str(forecast.get_error_rmse(true_y_future)))
    file.write("\n")
    file.close()
