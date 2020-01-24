import get_data
import joblib
import matplotlib.pyplot as plot
import numpy as np
from scipy.signal import find_peaks
from pandas.plotting import register_matplotlib_converters
from tabulate import tabulate

import pdb

def main():
    plot_length = 365
    time_array = get_data.get_time_test()
    time_array = time_array[0:plot_length]
    register_matplotlib_converters()
    
    forecast = joblib.load("../figures/london/mcmc/forecast.zlib")
    forecast_array = []
    for simulation in forecast.forecast_array:
        forecast_array.append(simulation.y_array[0:plot_length])
    forecast_array = np.asarray(forecast_array)
    n_simulation = len(forecast.forecast_array)
    
    training_rain = forecast.forecast_array[0].y_array
    y_extreme = np.quantile(training_rain, 0.99)
    y_observed = np.asarray(get_data.get_london_rain_test())
    y_observed = y_observed[0:plot_length]
    
    p_extreme = np.sum(forecast_array > y_extreme, 0) / n_simulation
    peaks, _ = find_peaks(p_extreme, height=0)
    extreme_day_forecast = np.zeros(plot_length, dtype=bool)
    extreme_day_forecast[peaks] = True
    
    warning_rain_array = [y_extreme, 20, 18, 15, 10]
    
    for warning_rain in warning_rain_array:
        warning_day = y_observed > warning_rain
        
        true_positive = (np.sum(np.logical_and(extreme_day_forecast, warning_day)))
        false_positive = np.sum(np.logical_and(extreme_day_forecast, np.logical_not(warning_day)))
        true_negative = np.sum(np.logical_and(np.logical_not(extreme_day_forecast), np.logical_not(warning_day)))
        false_negative = np.sum(np.logical_and(np.logical_not(extreme_day_forecast), warning_day))
        
        warning_rain_string = str(round(warning_rain))+" mm"
        table = [
            ["", "Rain < " + warning_rain_string, "Rain > " + warning_rain_string, ""],
            ["Forecast not extreme", true_negative, false_negative, true_negative+false_negative],
            ["Forecast extreme", false_positive, true_positive, false_positive+true_positive],
            ["", true_negative+false_positive, false_negative+true_positive, true_positive+false_positive+true_negative+false_negative],
        ]
        file = open("../figures/london/extreme"+str(round(warning_rain))+".txt","w")
        file.write(tabulate(table, headers="firstrow", tablefmt="latex"))
        file.close()
        
        plot.figure()
        plot.plot(time_array, p_extreme, label=r"$\hat{P}(rain>"+str(round(y_extreme, 2))+"mm)$")
        plot.plot(time_array[peaks], np.zeros_like(time_array[peaks]), '*k')
        for day in range(plot_length):
            if warning_day[day]:
                plot.axvline(x=time_array[day], color="r")
        plot.xlabel("time")
        plot.ylabel("forecasted probability of extreme rain")
        plot.legend()
        plot.savefig("../figures/london/extreme"+str(round(warning_rain))+".pdf")
        plot.show()

if __name__ == "__main__":
    main()
