import compound_poisson as cp
import joblib
from get_simulation import simulate_training

def main():
    time_series = simulate_training(cp.TimeSeriesSgd)
    time_series.initalise_parameters()
    time_series.fit()
    joblib.dump(time_series, "results/simulate_sgd.zlib")

if __name__ == "__main__":
    main()
