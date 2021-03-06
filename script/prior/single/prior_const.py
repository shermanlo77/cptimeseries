import os
from os import path
import pathlib

import numpy as np
from numpy import random

import prior_simulator

def main():
    
    rng = random.RandomState(np.uint32(3897768090))
    
    path_here = pathlib.Path(__file__).parent.absolute()
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    figure_dir = path.join(figure_dir, "const")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    
    prior_simulate = prior_simulator.time_series.PriorConstSimulator(
        figure_dir, rng)
    prior_simulate()

if __name__ == "__main__":
    main()
 
