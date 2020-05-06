import argparse

from numpy import random

import dataset
import fit_time_series

def main():
    parser = argparse.ArgumentParser(description="Sample size")
    parser.add_argument("--sample", help="number of mcmc samples", type=int)
    n_sample = parser.parse_args().sample

    seed = random.SeedSequence(170300509484813619611218577657545000221)
    fitter = fit_time_series.FitterSlice()
    fitter.fit(dataset.LondonSimulated(), seed, n_sample)

if __name__ == "__main__":
    main()
