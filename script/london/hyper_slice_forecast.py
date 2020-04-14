from os import path
import pathlib

import joblib

import compound_poisson
import dataset

def main():

    name = "hyper"

    path_here = pathlib.Path(__file__).parent.absolute()
    figure_dir = path.join(path_here, "figure", name)
    result_dir = path.join(path_here, "result")
    result_file = path.join(result_dir, name + ".gz")

    london = dataset.London80()
    n_simulation = 1000
    time_series = joblib.load(result_file)
    time_series.read_memmap()
    time_series.burn_in = 8000

    forecast_file = path.join(result_dir, name + "_forecast_training.gz")
    if not path.isfile(forecast_file):
        forecast = time_series.forecast_self(n_simulation)
        joblib.dump(forecast, forecast_file)
    else:
        forecast = joblib.load(forecast_file)
    rain = london.get_rain_training()
    compound_poisson.print.forecast(
        forecast, rain, figure_dir, "training")

    forecast_file = path.join(result_dir, name + "_forecast_test.gz")
    if not path.isfile(forecast_file):
        forecast = time_series.forecast(
            london.get_model_field_test(), n_simulation)
        joblib.dump(forecast, forecast_file)
    else:
        forecast = joblib.load(forecast_file)
    rain = london.get_rain_test()
    compound_poisson.print.forecast(forecast, rain, figure_dir, "test")

if __name__ == "__main__":
    main()
