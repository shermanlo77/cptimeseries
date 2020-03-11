import pathlib

import joblib
import numpy as np
from numpy import random

import compound_poisson
import dataset

def main():
    
    rng = random.RandomState(np.uint32(1919099529))
    downscale = compound_poisson.Downscale(dataset.IsleOfManTraining(), (5, 5))
    downscale.set_rng(rng)
    downscale.fit()
    
    path_here = pathlib.Path(__file__).parent.absolute()
    joblib.dump(downscale, path.join(path_here, "downscale.gz"))

if __name__ == "__main__":
    main()
