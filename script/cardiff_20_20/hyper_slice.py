from numpy import random

from compound_poisson import fit
from compound_poisson.fit import wrapper
import dataset


def main():
    fitter = fit.time_series.FitterHyperSlice()
    training = dataset.CardiffTraining()
    seed = random.SeedSequence(80188344912064343414862752267182073625)
    wrapper.time_series_fit(fitter, training, seed)


if __name__ == "__main__":
    main()
