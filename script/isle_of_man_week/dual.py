import pathlib

from numpy import random

from compound_poisson import multiprocess
import dataset
import fit_downscale

def main():
    pool = multiprocess.Pool()
    seed = random.SeedSequence(70599994716119404436100749277178204047)
    name = "dual"
    path_here = pathlib.Path(__file__).parent.absolute()
    fitter = fit_downscale.FitterDownscaleDual(name, path_here, pool, seed)
    fitter.fit(dataset.IsleOfManWeekTraining())
    pool.join()

if __name__ == "__main__":
    main()
