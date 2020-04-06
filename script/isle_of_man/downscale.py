from os import path
import pathlib

import joblib
import numpy as np
from numpy import random

import compound_poisson
import dataset

def main():

    seed = random.SeedSequence(41597761383904719560264433323691455830)
    downscale = compound_poisson.Downscale(dataset.IsleOfManTraining(), (5, 5))
    downscale.set_rng(seed)
    downscale.fit()

    path_here = pathlib.Path(__file__).parent.absolute()
    joblib.dump(downscale, path.join(path_here, "downscale.gz"))

if __name__ == "__main__":
    main()
