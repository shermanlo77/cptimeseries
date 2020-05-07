import argparse

from numpy import random

from compound_poisson import fit
import dataset

def main():
    parser = argparse.ArgumentParser(description="Sample size")
    parser.add_argument("--sample", help="number of mcmc samples", type=int)
    n_sample = parser.parse_args().sample

    seed = random.SeedSequence(199862391501461976584157354151760167878)
    fitter = fit.time_series.FitterMcmc()
    fitter.fit(dataset.LondonSimulated(), seed, n_sample)

if __name__ == "__main__":
    main()
