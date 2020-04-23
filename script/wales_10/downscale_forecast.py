import pathlib

from compound_poisson import multiprocess
import dataset
import fit_downscale

def main():
    pool = multiprocess.Pool()
    name = "downscale"
    path_here = pathlib.Path(__file__).parent.absolute()
    fitter = fit_downscale.FitterDownscale(name, path_here, pool)
    fitter.forecast(dataset.Wales10Test(), 1000, 200)
    pool.join()

if __name__ == "__main__":
    main()
