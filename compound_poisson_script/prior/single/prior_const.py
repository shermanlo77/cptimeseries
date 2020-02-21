import os
import sys

import numpy as np
import numpy.random as random

sys.path.append("..")
from prior_simulator import PriorConstSimulator

def main():
    
    rng = random.RandomState(np.uint32(3897768090))
    if not os.path.isdir("figure"):
        os.mkdir("figure")
    prior_simulator = PriorConstSimulator(os.path.join("figure", "const"), rng)
    prior_simulator()

if __name__ == "__main__":
    main()
 
