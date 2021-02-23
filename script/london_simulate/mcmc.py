from numpy import random

from compound_poisson import fit
from compound_poisson.fit import wrapper
import dataset

def main():
    fitter = fit.time_series.FitterMcmc()
    training = dataset.LondonSimulatedTraining()
    seed = random.SeedSequence(199862391501461976584157354151760167878)
    wrapper.time_series_fit(fitter, training, seed)

if __name__ == "__main__":
    main()
