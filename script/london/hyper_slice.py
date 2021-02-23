from numpy import random

from compound_poisson import fit
from compound_poisson.fit import wrapper
import dataset

def main():
    fitter = fit.time_series.FitterHyperSlice()
    training = dataset.LondonTraining()
    seed = random.SeedSequence(126906591942422578422472743313642430795)
    wrapper.time_series_fit(fitter, training, seed)

if __name__ == "__main__":
    main()
