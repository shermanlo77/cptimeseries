from numpy import random

from compound_poisson import fit
from compound_poisson.fit import wrapper
import dataset

def main():
    fitter = fit.time_series.FitterSlice()
    training = dataset.LondonSimulatedTraining()
    seed = random.SeedSequence(170300509484813619611218577657545000221)
    wrapper.time_series_fit(fitter, training, seed)

if __name__ == "__main__":
    main()
