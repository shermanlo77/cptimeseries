"""Compare the distribtuions of CP-MCMC, ERA5 with the observed, all together
    in the game graphs. For Wales.

Plotted:
    -survival plots
    -pp plots
    -qq plots
"""

import os
from os import path

import cycler
import joblib
from matplotlib import pyplot as plt

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

    observed_data = dataset.WalesTest()
    era5_data = dataset.Era5Wales()

    era5 = compound_poisson.era5.Downscale(era5_data)
    era5.fit(era5_data, observed_data)

    dir = path.join("..", "wales_5_20")
    downscale = joblib.load(
        path.join(dir, "result", "Downscale.gz"))
    downscale_name = "CP-MCMC (5)"
    era5_name = "IFS"
    observed_name = "observed"
    old_dir = downscale.forecaster.memmap_path
    downscale.forecaster.memmap_path = path.join(dir, old_dir)

    for time_series in downscale.generate_unmask_time_series():
        forecaster = time_series.forecaster
        old_dir = forecaster.memmap_path
        forecaster.memmap_path = path.join(dir, old_dir)

    downscale.forecaster.load_memmap("r")
    downscale.forecaster.load_locations_memmap("r")

    cp_comparer = downscale.forecaster.compare_dist_with_observed()
    era5_comparer = era5.forecaster.compare_dist_with_observed()

    #survival plot
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(monochrome)
    cp_comparer.plot_survival_forecast(downscale_name)
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
    cp_comparer.plot_pp(downscale_name)
    era5_comparer.plot_pp(era5_name)
    cp_comparer.adjust_pp_plot()
    plt.legend()
    plt.savefig(path.join(directory, "pp.pdf"), bbox_inches="tight")
    plt.close()

    #qq plot
    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(monochrome)
    cp_comparer.plot_qq(downscale_name)
    cp_comparer.adjust_qq_plot()
    era5_comparer.plot_qq(era5_name)
    plt.legend()
    plt.savefig(path.join(directory, "qq.pdf"), bbox_inches="tight")
    plt.close()

    downscale.forecaster.del_locations_memmap()
    downscale.forecaster.del_memmap()

if __name__ == "__main__":
    main()
