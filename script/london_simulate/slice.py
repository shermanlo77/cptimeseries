import argparse

from numpy import random

from compound_poisson import fit
import dataset

def main():
    parser = argparse.ArgumentParser(description="Sample size")
    parser.add_argument("--sample", help="number of mcmc samples", type=int)
    n_sample = parser.parse_args().sample

    seed = random.SeedSequence(170300509484813619611218577657545000221)
    fitter = fit.time_series.FitterSlice()
    fitter.fit(dataset.LondonSimulatedTraining(), seed, n_sample)

if __name__ == "__main__":
    main()
