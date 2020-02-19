import os
import sys

import joblib
import matplotlib.pyplot as plot
import numpy as np
import numpy.random as random

sys.path.append(os.path.join("..", ".."))
import compound_poisson as cp
import dataset

def main():
    
    rng = random.RandomState(np.uint32(3391431824))
    
    try:
        os.mkdir("result")
    except(FileExistsError):
        pass
    
    london = dataset.LondonSimulated80()
    model_field, rain = london.get_data_training()
    time_series = cp.TimeSeriesMcmc(model_field, rain, (5, 5), (5, 5))
    time_series.time_array = london.get_time_training()
    time_series.rng = rng
    time_series.fit()
    joblib.dump(time_series, os.path.join("result", "mcmc.gz"))

if __name__ == "__main__":
    main()
