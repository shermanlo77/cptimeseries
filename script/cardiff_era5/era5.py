import os
from os import path

import dataset
import compound_poisson

def main():

    directory = "figure"
    if not path.isdir(directory):
        os.mkdir(directory)

    era5 = dataset.Era5Cardiff()
    time_series = compound_poisson.era5.TimeSeries()
    time_series.fit(era5)

    observed_data = dataset.CardiffTest()

    compound_poisson.forecast.print.time_series(
        time_series.forecaster, observed_data.rain, directory, "test")

if __name__ == "__main__":
    main()
