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
    london = get_data.London2()
    x = london.get_model_field_training()
    rainfall = london.get_rain_training()
    model_field_name = x.columns
    x = np.asarray(x)
    time_series = cp.TimeSeriesMcmc(x, rainfall, (5,10), (5,10))
    time_series.model_field_name = model_field_name
    time_series.time_array = london.get_time_training()
    time_series.fit()
    joblib.dump(time_series, "results/mcmc_london2.zlib")

if __name__ == "__main__":
    main()
