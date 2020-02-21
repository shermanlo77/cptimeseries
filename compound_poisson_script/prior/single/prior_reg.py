import os
import sys

import numpy as np
import numpy.random as random

sys.path.append("..")
from prior_simulator import PriorRegSimulator

def main():
    
    rng = random.RandomState(np.uint32(2099961776))
    if not os.path.isdir("figure"):
        os.mkdir("figure")
    prior_simulator = PriorRegSimulator(os.path.join("figure", "reg"), rng)
    prior_simulator()

if __name__ == "__main__":
    main()
 
