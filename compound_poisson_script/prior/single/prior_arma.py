import os
import sys

import numpy as np
import numpy.random as random

sys.path.append("..")
from prior_simulator import PriorArmaSimulator

def main():
    
    rng = random.RandomState(np.uint32(2796796019))
    if not os.path.isdir("figure"):
        os.mkdir("figure")
    prior_simulator = PriorRegSimulator(os.path.join("figure", "arma"), rng)
    prior_simulator()

if __name__ == "__main__":
    main()
 
