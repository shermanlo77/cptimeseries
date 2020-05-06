import argparse
import pathlib

import joblib

import dataset
import fit_time_series

def main():
    parser = argparse.ArgumentParser(description="Forecasting options")
    parser.add_argument("--sample", help="number of simulations", type=int)
    parser.add_argument("--burnin", help="burn in", type=int)
    n_simulation = parser.parse_args().sample
    burn_in = parser.parse_args().burnin
    if burn_in is None:
        burn_in = 8000

    path_here = pathlib.Path(__file__).parent.absolute()
    fitter = fit_time_series.FitterHyperSlice(path_here)
    fitter.forecast(dataset.London80(), n_simulation, burn_in)

if __name__ == "__main__":
    main()
