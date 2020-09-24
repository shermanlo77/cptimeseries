import argparse

import joblib

from compound_poisson import fit
import dataset

def main():
    parser = argparse.ArgumentParser(description="Forecasting options")
    parser.add_argument("--sample", help="number of simulations", type=int)
    parser.add_argument("--burnin", help="burn in", type=int)
    n_simulation = parser.parse_args().sample
    burn_in = parser.parse_args().burnin
    if burn_in is None:
        burn_in = 40000

    fitter = fit.time_series.FitterHyperSlice()
    fitter.forecast((dataset.CardiffTraining(), dataset.CardiffTest()),
                    n_simulation,
                    burn_in)

if __name__ == "__main__":
    main()
