import os
import sys

import joblib
import numpy as np
from numpy.random import RandomState

sys.path.append(os.path.join("..", ".."))
import dataset

def main():
    
    downscale = joblib.load("downscale.gz")
    test_set = dataset.IsleOfManTest()
    forecast = downscale.forecast(test_set, 1000)
    joblib.dump(forecast, "forecast.gz")

if __name__ == "__main__":
    main()
