import argparse

from numpy import random

from compound_poisson import fit
from compound_poisson import multiprocess
import dataset

def main():
    pool = multiprocess.MPIPoolExecutor()

    parser = argparse.ArgumentParser(description="Sample size")
    parser.add_argument("--sample", help="number of mcmc samples", type=int)
    n_sample = parser.parse_args().sample

    seed = random.SeedSequence(96402363659177019578667127138226980756)
    fitter = fit.downscale.FitterDownscale()
    fitter.fit(dataset.Wales10Test(), seed, n_sample, pool)

    pool.join()

if __name__ == "__main__":
    main()
