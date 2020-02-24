import os
import sys

import numpy as np
import numpy.random as random

sys.path.append("..")
sys.path.append(os.path.join("..", ".."))
import dataset
from fitter import FitterHyperSlice

def main():
    
    rng = random.RandomState(np.uint32(2443707582))
    name = "hyper"
    result_dir = "result"
    figure_dir = "figure"
    fitter = FitterHyperSlice(
        dataset.London80(), rng, name, result_dir, figure_dir)
    fitter()

if __name__ == "__main__":
    main()
