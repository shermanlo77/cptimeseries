import pathlib

from numpy import random

from compound_poisson import multiprocess
import dataset
import fit_downscale

def main():
    pool = multiprocess.BackendMPI()
    seed = random.SeedSequence(306149477262471971409074842221838773037)
    name = "downscale"
    path_here = pathlib.Path(__file__).parent.absolute()
    fitter = fit_downscale.FitterDownscale(name, path_here, pool, seed)
    fitter.fit(dataset.AnaDual1Training())
    pool.join()

if __name__ == "__main__":
    main()
