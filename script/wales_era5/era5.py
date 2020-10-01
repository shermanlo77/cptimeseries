import os
from os import path

import dataset
import compound_poisson
from compound_poisson import multiprocess

def main():

    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)
    pool = multiprocess.Pool()

    observed_data = dataset.CardiffTest()

    era5 = dataset.Era5Wales()
    downscale = compound_poisson.era5.Downscale(era5)
    downscale.fit(era5, observed_data)

    compound_poisson.forecast.print.downscale(
        downscale.forecaster, observed_data, directory, pool)

if __name__ == "__main__":
    main()
