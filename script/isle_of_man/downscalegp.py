from numpy import random

from compound_poisson import fit
from compound_poisson import multiprocess
from compound_poisson.fit import wrapper
import dataset

def main():
    fitter = fit.downscale.FitterDownscaleDeepGp()
    data = dataset.IsleOfManTraining()
    seed = random.SeedSequence(335181766240425557327571375931666354614)
    Pool = multiprocess.Pool
    wrapper.downscale_fit(fitter, data, seed, Pool)

if __name__ == "__main__":
    main()
