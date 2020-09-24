import os
from os import path

import dataset
import compound_poisson
from compound_poisson import multiprocess

def main():

    pool = multiprocess.Pool()
    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)

    data = dataset.Era5IsleOfMan()
    downscale = compound_poisson.era5.Downscale(data)
    test_set = dataset.IsleOfManTest()
    downscale.fit(data, test_set)
    compound_poisson.forecast.print.downscale(
        downscale.forecaster, test_set, directory, pool)

if __name__ == "__main__":
    main()
