from compound_poisson import fit
from compound_poisson import multiprocess
from compound_poisson.fit import wrapper
import dataset


def main():
    fitter = fit.downscale.FitterMultiSeries()
    test = dataset.WalesTest()
    default_burn_in = 15000
    Pool = multiprocess.Pool
    wrapper.downscale_forecast(fitter, test, default_burn_in, Pool)


if __name__ == "__main__":
    main()
