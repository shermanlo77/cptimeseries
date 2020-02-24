import os
import sys

import numpy as np
import numpy.random as random

sys.path.append("..")
sys.path.append(os.path.join("..", ".."))
import dataset
from fitter import FitterMcmc

def main():
    
    rng = random.RandomState(np.uint32(3391431824))
    name = "mcmc"
    result_dir = "result"
    figure_dir = "figure"
    fitter = FitterMcmc(
        dataset.LondonSimulated80(), rng, name, result_dir, figure_dir)
    fitter()

if __name__ == "__main__":
    main()
