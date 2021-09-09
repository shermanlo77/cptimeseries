from numpy import random

from compound_poisson import fit
from compound_poisson import multiprocess
from compound_poisson.fit import wrapper
import dataset


def main():
    fitter = fit.downscale.FitterMultiSeries(suffix="gp")
    data = dataset.Wales5Training()
    seed = random.SeedSequence(336116686577838597869553922167649360230)
    Pool = multiprocess.Pool
    wrapper.downscale_fit(fitter, data, seed, Pool)


if __name__ == "__main__":
    main()
