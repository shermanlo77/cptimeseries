import argparse

from compound_poisson import fit
from compound_poisson import multiprocess
import dataset

def main():
    pool = multiprocess.Pool()

    parser = argparse.ArgumentParser(description="Forecasting options")
    parser.add_argument("--sample", help="number of simulations", type=int)
    parser.add_argument("--burnin", help="burn in", type=int)
    n_simulation = parser.parse_args().sample
    burn_in = parser.parse_args().burnin
    if burn_in is None:
        burn_in = 200

    fitter = fit.downscale.FitterDownscale()
    fitter.forecast(dataset.IsleOfManTest(), n_simulation, burn_in, pool)
    pool.join()

if __name__ == "__main__":
    main()
