import compound_poisson as cp
from get_data import LondonSimulation
import joblib

def main():
    london = LondonSimulation()
    x = london.get_model_field_training()
    rainfall = london.get_rain_training()
    time_series = cp.TimeSeriesMcmc(x, rainfall, (2, 2), (2, 2))
    time_series.time_array = london.get_time_training()
    time_series.fit()
    joblib.dump(time_series, "results/london/simulation/mcmc.zlib")

if __name__ == "__main__":
    main()
