import os
from os import path
import pathlib

import matplotlib.pyplot as plt

import compound_poisson
import dataset

def main():

    path_here = pathlib.Path(__file__).parent.absolute()
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)

    london = dataset.London()
    model_field = london.model_field
    rain = london.rain
    time_series = compound_poisson.TimeSeries(model_field, rain)
    time_series.time_array = london.time_array
    compound_poisson.print.time_series(time_series, figure_dir)

    for i in range(time_series.n_model_field):
        plt.figure()
        plt.plot(time_series.time_array, time_series.x[:, i])
        plt.xlabel("Time")
        plt.ylabel(time_series.model_field_name[i])
        plt.savefig(path.join(figure_dir, "model_field_" + str(i) + ".pdf"))
        plt.close()

if __name__ == "__main__":
    main()
