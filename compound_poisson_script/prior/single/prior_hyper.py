import os
import sys

import numpy as np
import numpy.random as random

sys.path.append("..")
from prior_simulator import PriorSimulator

def main():
    
    rng = random.RandomState(np.uint32(2369071836))
    if not os.path.isdir("figure"):
        os.mkdir("figure")
    prior_simulator = PriorSimulator(os.path.join("figure", "hyper"), rng)
    prior_simulator()

if __name__ == "__main__":
    main()
 
