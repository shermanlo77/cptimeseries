import compound_poisson as cp
from get_data import London
import joblib

def main():
    london = London()
    x = london.get_model_field_training()
    rainfall = london.get_rain_training()
    time_series = cp.TimeSeriesMcmc(x, rainfall, (10, 5), (10, 5))
    time_series.time_array = london.get_time_training()
    time_series.fit()
    joblib.dump(time_series, "results/london/mcmc.zlib")

if __name__ == "__main__":
    main()
