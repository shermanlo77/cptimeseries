from numpy import random

from compound_poisson import fit
from compound_poisson import multiprocess
from compound_poisson.fit import wrapper
import dataset

def main():
    fitter = fit.downscale.FitterDownscale()
    data = dataset.IsleOfManTraining()
    seed = random.SeedSequence(41597761383904719560264433323691455830)
    Pool = multiprocess.Pool
    wrapper.downscale_fit(fitter, data, seed, Pool)

if __name__ == "__main__":
    main()
