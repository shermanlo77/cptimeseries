import argparse

from numpy import random

import dataset
import fit_time_series

def main():
    parser = argparse.ArgumentParser(description="Sample size")
    parser.add_argument("--sample", help="number of mcmc samples", type=int)
    n_sample = parser.parse_args().sample

    seed = random.SeedSequence(126906591942422578422472743313642430795)
    fitter = fit_time_series.FitterHyperSlice()
    fitter.fit(dataset.London(), seed, n_sample)

if __name__ == "__main__":
    main()
