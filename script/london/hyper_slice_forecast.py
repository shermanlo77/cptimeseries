import argparse
from os import path
import pathlib

import joblib

import compound_poisson
import dataset

def main():

    name = "hyper"

    parser = argparse.ArgumentParser(description="Forecasting options")
    parser.add_argument("--sample", help="number of simulations", type=int)
    n_simulation = parser.parse_args().sample
    if n_simulation is None:
        n_simulation = 1000

    path_here = pathlib.Path(__file__).parent.absolute()
    figure_dir = path.join(path_here, "figure", name)
    result_dir = path.join(path_here, "result")
    result_file = path.join(result_dir, name + ".gz")

    london = dataset.London80()
    time_series = joblib.load(result_file)
    time_series.burn_in = 8000
    time_series.forecaster_memmap_dir = result_dir

    self_forecaster = time_series.forecast_self(n_simulation)
    forecaster = time_series.forecast(
        london.get_model_field_test(), n_simulation)
    joblib.dump(time_series, result_file)

    rain = london.get_rain_training()
    compound_poisson.print.forecast(
        self_forecaster, rain, figure_dir, "training")

    rain = london.get_rain_test()
    compound_poisson.print.forecast(forecaster, rain, figure_dir, "test")

if __name__ == "__main__":
    main()
