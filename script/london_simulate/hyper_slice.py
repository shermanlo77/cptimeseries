import os
from os import path
import pathlib

import numpy as np
from numpy import random

import dataset
import fitter

def main():
    
    rng = random.RandomState(np.uint32(3855346694))
    name = "hyper"
    
    path_here = pathlib.Path(__file__).parent.absolute()
    
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    
    result_dir = path.join(path_here, "result")
    if not path.isdir(result_dir):
        os.mkdir(result_dir)
    
    fit = fitter.FitterHyperSlice(
        dataset.LondonSimulated80(), rng, name, result_dir, figure_dir)
    fit()

if __name__ == "__main__":
    main()
