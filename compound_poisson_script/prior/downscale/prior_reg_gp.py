import os
import sys

import numpy as np
import numpy.random as random

sys.path.append("..")
from prior_simulator import PriorDsRegSimulator

def main():
    
    rng = random.RandomState(np.uint32(4187205155))
    if not os.path.isdir("figure"):
        os.mkdir("figure")
    prior_simulator = PriorDsRegSimulator(os.path.join("figure", "reg_gp"), rng)
    prior_simulator()

if __name__ == "__main__":
    main()
 
