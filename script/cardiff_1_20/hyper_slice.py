from numpy import random

from compound_poisson import fit
from compound_poisson.fit import wrapper
import dataset


def main():
    fitter = fit.time_series.FitterHyperSlice()
    training = dataset.Cardiff1Training()
    seed = random.SeedSequence(277310809467192855312273294721104678816)
    wrapper.time_series_fit(fitter, training, seed)


if __name__ == "__main__":
    main()
