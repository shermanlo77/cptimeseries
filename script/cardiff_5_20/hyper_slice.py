from numpy import random

from compound_poisson import fit
from compound_poisson.fit import wrapper
import dataset

def main():
    fitter = fit.time_series.FitterHyperSlice()
    training = dataset.Cardiff5Training()
    seed = random.SeedSequence(230692462564320493984147630542548799902)
    wrapper.time_series_fit(fitter, training, seed)

if __name__ == "__main__":
    main()
