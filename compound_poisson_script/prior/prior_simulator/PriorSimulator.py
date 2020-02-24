import os
import sys

from cycler import cycler
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import statsmodels.tsa.stattools as stats

import compound_poisson as cp
from dataset import Ana_1

class PriorSimulator:
    
    def __init__(self, figure_directory, rng):
        self.figure_directory = figure_directory
        self.rng = rng
        self.n_arma = (5, 5)
        self.n_simulate = 100
        self.n_lag = 20
        self.max_rain = 100
        self.data = Ana_1()
        if not os.path.isdir(figure_directory):
            os.mkdir(figure_directory)
    
    def simulate_prior_time_series(self, x, prior_std):
        time_series = cp.TimeSeriesMcmc(
            x, poisson_rate_n_arma=self.n_arma,
            gamma_mean_n_arma=self.n_arma)
        time_series.rng = self.rng
        time_series.simulate_from_prior()
        return time_series
    
    def simulate_hyper_time_series(self, x):
        time_series = cp.TimeSeriesHyperSlice(
            x, poisson_rate_n_arma=self.n_arma,
            gamma_mean_n_arma=self.n_arma)
        time_series.rng = self.rng
        time_series.simulate_from_prior()
        return time_series
    
    def get_time_series(self, prior_std=None):
        x = self.data.get_model_field_random(self.rng)
        if prior_std is None:
            time_series = self.simulate_hyper_time_series(x)
        else:
            time_series = self.simulate_prior_time_series(x, prior_std)
        time_series.time_array = self.data.time_array
        return time_series
    
    def print(self, figure_directory=None, prior_std=None):
        
        if figure_directory is None:
            figure_directory = self.figure_directory
        if not os.path.isdir(figure_directory):
            os.mkdir(figure_directory)
        
        acf_array = []
        pacf_array = []
        rain_sorted_array = []
        
        for i_simulate in range(self.n_simulate):
            time_series = self.get_time_series(prior_std)
            cp.print.time_series(
                time_series, figure_directory, str(i_simulate) + "_")
            y = time_series.y_array
            acf = stats.acf(y, nlags=self.n_lag, fft=True)
            try:
                pacf = stats.pacf(y, nlags=self.n_lag)
            except(stats.LinAlgError):
                pacf = np.full(self.n_lag + 1, np.nan)
            if (len(acf) == self.n_lag + 1) and (not np.any(np.isnan(acf))):
                acf_array.append(acf)
                pacf_array.append(pacf)
                rain_sorted_array.append(np.sort(y))
            
        colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        cycle = cycler(color=[colours[0]],
                                linewidth=[0.5],
                                alpha=[0.25])
        n = len(time_series)
        cdf = np.asarray(range(n))
        plot.figure()
        ax = plot.gca()
        ax.set_prop_cycle(cycle)
        for y in rain_sorted_array:
            if y[-1] < self.max_rain:
                plot.plot(np.concatenate(([0], y, [self.max_rain])),
                          np.concatenate(([0], cdf, [n])))
            else:
                plot.plot(np.concatenate(([0], y)), np.concatenate(([0], cdf)))
            if np.any(y == 0):
                non_zero_index = np.nonzero(y)[0][0] - 1
                plot.scatter(0, cdf[non_zero_index], alpha=0.25)
        plot.xlim(0, self.max_rain)
        plot.xlabel("rainfall (mm)")
        plot.ylabel("cumulative frequency")
        plot.savefig(os.path.join(figure_directory, "prior_cdf.pdf"))
        plot.close()
        
        acf = np.asarray(acf_array)
        plot.figure()
        plot.boxplot(acf[:,1:self.n_lag+1])
        plot.xlabel("lag (day)")
        plot.ylabel("autocorrelation")
        plot.savefig(os.path.join(figure_directory, "prior_acf.pdf"))
        plot.close()
        
        pacf = np.asarray(pacf_array)
        plot.figure()
        plot.boxplot(pacf[:,1:self.n_lag+1])
        plot.xlabel("lag (day)")
        plot.ylabel("partial autocorrelation")
        plot.savefig(os.path.join(figure_directory, "prior_pacf.pdf"))
        plot.close()
    
    def __call__(self):
        self.print()
