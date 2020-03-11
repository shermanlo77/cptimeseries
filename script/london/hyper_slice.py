import os
from os import path

import numpy as np
from numpy import random
import pathlib

import dataset
import fitter

def main():
    
    rng = random.RandomState(np.uint32(2443707582))
    name = "hyper"
    
    path_here = pathlib.Path(__file__).parent.absolute()
    
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    
    result_dir = path.join(path_here, "result")
    if not path.isdir(result_dir):
        os.mkdir(result_dir)
    
    fit = FitterHyperSlice(
        dataset.London80(), rng, name, result_dir, figure_dir)
    fit()

if __name__ == "__main__":
    main()
