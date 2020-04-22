import argparse
import pathlib

from numpy import random

from compound_poisson import multiprocess
import dataset
import fit_downscale

def main():
    pool = multiprocess.BackendMPI()

    parser = argparse.ArgumentParser(description="Sample size")
    parser.add_argument("--sample", help="number of mcmc samples", type=int)
    n_sample = parser.parse_args().sample

    seed = random.SeedSequence(257590222684807240610971838917335796260)
    name = "downscale"
    path_here = pathlib.Path(__file__).parent.absolute()
    fitter = fit_downscale.FitterDownscale(name, path_here, pool, seed)
    fitter.fit(dataset.AnaDual10Training(), n_sample)
    pool.join()

if __name__ == "__main__":
    main()
