from numpy import random

from compound_poisson import fit
from compound_poisson.fit import wrapper
import dataset

def main():
    fitter = fit.time_series.FitterHyperSlice()
    training = dataset.Cardiff10Training()
    seed = random.SeedSequence(177782466634943011322205683796258167716)
    wrapper.time_series_fit(fitter, training, seed)

if __name__ == "__main__":
    main()
