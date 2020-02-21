import os
import sys

import joblib
import matplotlib.pyplot as plot
import numpy as np
import numpy.random as random

sys.path.append(os.path.join("..", ".."))
import compound_poisson as cp
import dataset

def main():
    
    name = "hyper"
    result_dir = "result"
    figure_directory = os.path.join("figure", name)
    result_file = os.path.join(result_dir, name + ".gz")
    london = dataset.London80()
    n_simulation = 100
    time_series = joblib.load(result_file)
    time_series.burn_in = 8000
    
    forecast_file = os.path.join(result_dir, name + "_forecast_training.gz")
    if not os.path.isfile(forecast_file):
        forecast = time_series.forecast_self(n_simulation)
        joblib.dump(forecast, forecast_file)
    else:
        forecast = joblib.load(forecast_file)
    rain = london.get_rain_training()
    cp.print.forecast(forecast, rain, figure_directory, "training")
    
    forecast_file = os.path.join(result_dir, name + "_forecast_test.gz")
    if not os.path.isfile(forecast_file):
        forecast = time_series.forecast(
            london.get_model_field_test(), n_simulation)
        joblib.dump(forecast, forecast_file)
    else:
        forecast = joblib.load(forecast_file)
    rain = london.get_rain_test()
    cp.print.forecast(forecast, rain, figure_directory, "test")

if __name__ == "__main__":
    main()
