import os
from os import path

import dataset
import compound_poisson
from compound_poisson.forecast import print


def main():

    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)

    era5 = dataset.Era5Cardiff()
    time_series = compound_poisson.era5.TimeSeries()
    time_series.fit(era5)

    observed_data = dataset.CardiffTest()

    printer = print.TimeSeries(
        time_series.forecaster, observed_data.rain, directory, "test")
    printer.print()


if __name__ == "__main__":
    main()
