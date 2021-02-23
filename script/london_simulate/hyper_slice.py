from numpy import random

from compound_poisson import fit
from compound_poisson.fit import wrapper
import dataset

def main():
    fitter = fit.time_series.FitterHyperSlice()
    training = dataset.LondonSimulatedTraining()
    seed = random.SeedSequence(294372210542946537575453307391036609937)
    wrapper.time_series_fit(fitter, training, seed)

if __name__ == "__main__":
    main()
