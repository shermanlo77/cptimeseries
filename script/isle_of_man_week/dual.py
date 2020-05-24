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

    seed = random.SeedSequence(70599994716119404436100749277178204047)
    fitter = fit.downscale.FitterDownscale()
    fitter.fit(dataset.IsleOfManTraining(), seed, n_sample, pool)

    pool.join()

if __name__ == "__main__":
    main()
