import argparse

from numpy import random

from compound_poisson import fit
from compound_poisson import multiprocess
import dataset

def main():
    pool = multiprocess.Pool()

    parser = argparse.ArgumentParser(description="Sample size")
    parser.add_argument("--sample", help="number of mcmc samples", type=int)
    n_sample = parser.parse_args().sample

    seed = random.SeedSequence(336116686577838597869553922167649360230)
    fitter = fit.downscale.FitterMultiSeries()
    fitter.fit(dataset.Wales5Training(), seed, n_sample, pool)

    pool.join()

if __name__ == "__main__":
    main()
