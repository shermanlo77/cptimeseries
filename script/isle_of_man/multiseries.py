from numpy import random

from compound_poisson import fit
from compound_poisson import multiprocess
from compound_poisson.fit import wrapper
import dataset

def main():
    fitter = fit.downscale.FitterMultiSeries()
    data = dataset.IsleOfManTraining()
    seed = random.SeedSequence(275033816910622348579815457010957489899)
    Pool = multiprocess.Pool
    wrapper.downscale_fit(fitter, data, seed, Pool)

if __name__ == "__main__":
    main()
