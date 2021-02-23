from numpy import random

from compound_poisson import fit
from compound_poisson import multiprocess
from compound_poisson.fit import wrapper
import dataset

def main():
    fitter = fit.downscale.FitterDownscaleDeepGp()
    data = dataset.Wales5Training()
    seed = random.SeedSequence(328639170479038110469527951353307569903)
    Pool = multiprocess.Pool
    wrapper.downscale_fit(fitter, data, seed, Pool)

if __name__ == "__main__":
    main()
