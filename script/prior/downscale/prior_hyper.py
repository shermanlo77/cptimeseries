import os
from os import path
import pathlib

import numpy as np
from numpy import random

import prior_simulator

def main():

    seed = random.SeedSequence(332301838246917065154383428780003278502)

    path_here = pathlib.Path(__file__).parent.absolute()
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    figure_dir = path.join(figure_dir, "hyper")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)

    prior_simulate = prior_simulator.downscale.PriorSimulator(figure_dir, seed)
    prior_simulate()

if __name__ == "__main__":
    main()
