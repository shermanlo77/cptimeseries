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

    rng = random.RandomState(np.uint32(2443707582))
    name = "hyper"
    path_here = pathlib.Path(__file__).parent.absolute()
    fitter = fit_time_series.FitterHyperSlice(name, path_here, rng)
    fitter.fit(dataset.London80(), n_sample)

if __name__ == "__main__":
    main()
