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

    seed = random.SeedSequence(323329356722452848150901783436673986519)
    fitter = fit.downscale.FitterDownscaleDual()
    fitter.fit(dataset.WalesTraining(), seed, n_sample, pool)

    pool.join()

if __name__ == "__main__":
    main()