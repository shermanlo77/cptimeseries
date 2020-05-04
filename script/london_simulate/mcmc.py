import argparse
import pathlib

import numpy as np
from numpy import random

import dataset
import fit_time_series

def main():
    parser = argparse.ArgumentParser(description="Sample size")
    parser.add_argument("--sample", help="number of mcmc samples", type=int)
    n_sample = parser.parse_args().sample

    seed = random.SeedSequence(199862391501461976584157354151760167878)
    path_here = pathlib.Path(__file__).parent.absolute()
    fitter = fit_time_series.FitterMcmc(path_here)
    fitter.fit(dataset.LondonSimulated80(), seed, n_sample)

if __name__ == "__main__":
    main()
