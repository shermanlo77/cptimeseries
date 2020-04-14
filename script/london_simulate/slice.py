import pathlib

import numpy as np
from numpy import random

import dataset
import fit_time_series

def main():
    rng = random.RandomState(np.uint32(3976111046))
    name = "slice"
    path_here = pathlib.Path(__file__).parent.absolute()
    fitter = fit_time_series.FitterSlice(name, path_here, rng)
    fitter.fit(dataset.LondonSimulated80())

if __name__ == "__main__":
    main()
