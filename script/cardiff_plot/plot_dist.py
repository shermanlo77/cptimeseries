import os
from os import path

import cycler
import joblib
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
import pandas.plotting

import compound_poisson
import dataset

LINESTYLE = ['-', '--', '-.', ':']

def main():

    monochrome = (cycler.cycler('color', ['k'])
        * cycler.cycler('linestyle', LINESTYLE))
    plt.rcParams.update({'font.size': 14})

    #where to save the figures
    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)

    era5 = compound_poisson.era5.TimeSeries()
    era5.fit(dataset.Era5Cardiff())

    observed_data = dataset.CardiffTest()
    observed_rain = observed_data.rain

    dir = path.join("..", "cardiff_5_20")
    time_series = joblib.load(
        path.join(dir, "result", "TimeSeriesHyperSlice.gz"))
    time_series_name = "CP-MCMC (5)"
    era5_name = "IFS"
    observed_name = "observed"
    old_dir = time_series.forecaster.memmap_path
    time_series.forecaster.memmap_path = path.join(dir, old_dir)
    time_series.forecaster.load_memmap("r")

    cp_comparer = time_series.forecaster.compare_dist_with_observed(
        observed_rain)
    era5_comparer = era5.forecaster.compare_dist_with_observed(
        observed_rain)

    #survival plot
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(monochrome)
    cp_comparer.plot_survival_forecast(time_series_name)
    era5_comparer.plot_survival_forecast(era5_name)
    era5_comparer.plot_survival_observed(observed_name)
    cp_comparer.adjust_survival_plot()
    plt.legend()
    plt.savefig(path.join(directory, "survival.pdf"), bbox_inches="tight")
    plt.close()

    #pp plot
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(monochrome)
    cp_comparer.plot_pp(time_series_name)
    era5_comparer.plot_pp(era5_name)
    cp_comparer.adjust_pp_plot()
    plt.legend()
    plt.savefig(path.join(directory, "pp.pdf"), bbox_inches="tight")
    plt.close()

    #qq plot
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(monochrome)
    cp_comparer.plot_qq(time_series_name)
    cp_comparer.adjust_qq_plot()
    era5_comparer.plot_qq(era5_name)
    plt.legend()
    plt.savefig(path.join(directory, "qq.pdf"), bbox_inches="tight")
    plt.close()

    time_series.forecaster.del_memmap()

if __name__ == "__main__":
    main()
