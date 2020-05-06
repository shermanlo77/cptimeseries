import argparse

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

    fitter = fit_time_series.FitterHyperSlice()
    fitter.forecast(dataset.London(), n_simulation, burn_in)

if __name__ == "__main__":
    main()
