import os
import sys

import joblib
import numpy as np
from numpy.random import RandomState

sys.path.append(os.path.join("..", ".."))
from compound_poisson import Downscale
from dataset import IsleOfManTraining

def main():
    
    rng = RandomState(np.uint32(1919099529))
    downscale = Downscale(IsleOfManTraining(), (5, 5))
    downscale.set_rng(rng)
    downscale.fit()
    joblib.dump(downscale, "downscale.gz")

if __name__ == "__main__":
    main()
