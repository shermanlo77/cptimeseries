import argparse

from compound_poisson import fit
from compound_poisson import multiprocess
import dataset

def main():
    pool = multiprocess.MPIPoolExecutor()

    parser = argparse.ArgumentParser(description="Forecasting options")
    parser.add_argument("--sample", help="number of simulations", type=int)
    parser.add_argument("--burnin", help="burn in", type=int)
    parser.add_argument('--noprint', default=False, action="store_true")

    n_simulation = parser.parse_args().sample
    burn_in = parser.parse_args().burnin
    if burn_in is None:
        burn_in = 3000
    is_print = not parser.parse_args().noprint

    fitter = fit.downscale.FitterDownscaleDual()
    fitter.forecast(dataset.WalesTest(), n_simulation, burn_in, pool, is_print)
    pool.join()

if __name__ == "__main__":
    main()
