#script exploratory analysis
#
#variables looked at: rainfall, model fields over space and time
#cities looked at: London, Cardiff, Edinburgh, Belfast, Dublin
#
#plot mean (over time) for each point in space (as a heat map)
#plot time series for each city (along with acf and pacf
#scatter plot yesterday and today rainfall for each city
#matrix plot of all the variables for each city

import compound_poisson as cp
import get_data
import joblib
import numpy as np

def main():
    x = get_data.get_london_model_field_training()
    rainfall = get_data.get_london_rain_training()
    model_field_name = x.columns
    x = np.asarray(x)
    time_series = cp.TimeSeriesMcmc(x, rainfall=rainfall)
    time_series.model_field_name = model_field_name
    time_series.time_array = get_data.get_time_training()
    time_series.fit()
    joblib.dump(time_series, "results/mcmc_london.zlib")

if __name__ == "__main__":
    main()
