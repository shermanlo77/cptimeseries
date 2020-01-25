import compound_poisson as cp
import joblib
from get_simulation import simulate_training

import matplotlib.pyplot as plot

def main():
    time_series = simulate_training(cp.TimeSeriesGd)
    time_series.initalise_parameters_given_arma()
    time_series.fit()
    
    plot.figure()
    plot.plot(time_series.ln_l_array)
    plot.show()

if __name__ == "__main__":
    main()
