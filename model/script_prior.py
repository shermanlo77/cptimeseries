import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import statsmodels.tsa.stattools as stats
from cycler import cycler

from get_data import London
from print_figure import print_time_series
from TimeSeriesMcmc import TimeSeriesMcmc

def main():
    london = London()
    model_field = london.get_model_field_training()
    rain = london.get_rain_training()
    time_series = TimeSeriesMcmc(model_field, rain, (5, 10), (5, 10))
    time_series.time_array = london.get_time_training()
    
    n_lag = 20
    n_simulation = 100
    max_rain = 100
    n = len(time_series)
    cdf = np.asarray(range(n))
    
    directory = "../figures/london/prior/"
    
    acf_array = []
    pacf_array = []
    rain_sorted_array = []
    
    for i in range(n_simulation):
        time_series.simulate_from_prior()
        print_time_series(
            time_series, directory + "prior_" + str(i) + "_")
        y = time_series.y_array
        acf = stats.acf(y, nlags=n_lag, fft=True)
        pacf = stats.pacf(y, nlags=n_lag)
        if (len(acf) == n_lag + 1) and (not np.any(np.isnan(acf))):
            acf_array.append(acf)
            pacf_array.append(pacf)
            rain_sorted_array.append(np.sort(y))
    
    colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    cycle = cycler(color=[colours[0]],
                            linewidth=[0.5],
                            alpha=[0.25])
    plot.figure()
    ax = plot.gca()
    ax.set_prop_cycle(cycle)
    for y in rain_sorted_array:
        if y[-1] < max_rain:
            plot.plot(
                np.concatenate((y, [max_rain])), np.concatenate((cdf, [n])))
        else:
            plot.plot(y, cdf)
        if np.any(y == 0):
            non_zero_index = np.nonzero(y)[0][0] - 1
            plot.scatter(0, cdf[non_zero_index], alpha=0.25)
    plot.xlim(0, max_rain)
    plot.xlabel("rainfall (mm)")
    plot.ylabel("cumulative frequency")
    plot.savefig(directory + "prior_cdf.pdf")
    plot.close()
    
    acf = np.asarray(acf_array)
    plot.figure()
    plot.boxplot(acf[:,1:n_lag+1])
    plot.xlabel("lag (day)")
    plot.ylabel("autocorrelation")
    plot.savefig(directory + "prior_acf.pdf")
    plot.close()
    
    pacf = np.asarray(pacf_array)
    plot.figure()
    plot.boxplot(pacf[:,1:n_lag+1])
    plot.xlabel("lag (day)")
    plot.ylabel("partial autocorrelation")
    plot.savefig(directory + "prior_pacf.pdf")
    plot.close()

if __name__ == "__main__":
    main()
