"""Plots the ROC curve with ERA5 as a point
"""

import os
from os import path

import joblib
import matplotlib.pyplot as plt
import numpy as np

import compound_poisson
import dataset

RAIN_ARRAY = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25]

def main():

    directory = "figure_roc_compare"
    if not path.isdir(directory):
        os.mkdir(directory)

    era5 = compound_poisson.era5.TimeSeries()
    era5.fit(dataset.Era5Cardiff())

    observed_data = dataset.CardiffTest()

    time_series = joblib.load(path.join("result", "TimeSeriesHyperSlice.gz"))
    time_series.forecaster.load_memmap("r")

    for rain in RAIN_ARRAY:
        positive_array = era5.forecaster.forecast > rain
        alt_array = observed_data.rain > rain
        null_array = np.logical_not(alt_array)

        true_positive_rate = (np.sum(np.logical_and(positive_array, alt_array))
            / np.sum(alt_array))
        false_positive_rate = (np.sum(
            np.logical_and(positive_array, null_array))
            / np.sum(null_array))

        roc = time_series.forecaster.get_roc_curve(rain, observed_data.rain)
        plt.figure()
        roc.plot()
        plt.scatter(false_positive_rate, true_positive_rate, label="ERA5")
        plt.legend()
        plt.savefig(path.join(directory, str(rain)+".pdf"))
        plt.close()

    time_series.forecaster.del_memmap()

if __name__ == "__main__":
    main()
