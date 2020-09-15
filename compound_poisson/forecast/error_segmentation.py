"""For plotting errors for every segmentation (eg, every year)

Plot the errors for each segmentation
Also plot (as a horizontal line) the error for all segmentations combined
"""

from os import path

import matplotlib.pyplot as plt
import pandas.plotting

from compound_poisson.forecast import error

#list of all the errors to plot
ERROR_CLASSES = [
    error.RootMeanSquareError,
    error.RootMeanSquare10Error,
    error.MeanAbsoluteError,
]

class TimeSeries(object):

    def plot_error(forecast,
                   observed_rain,
                   time_segmentator,
                   directory,
                   prefix=""):
        """
        Args:
            forecast: Forecaster object
            observed_rain: numpy array of observed precipitation
            time_segmentator: TimeSegmenator object
            directory: where to save the figure
            prefix: what to name the figure
        """

        time_array = [] #array of dates for each segmentation
        #array of error objects when combining the segmentations
        error_all_array = []
        #array of arrays, for each error, each containing error for each
            #segmentation
        error_plot_array = []

        #init error objects and variables
        for error_class in ERROR_CLASSES:
            error_all_array.append(error_class())
            error_plot_array.append([])

        #for each segmentation
        for date, index in time_segmentator:
            time_array.append(date) #get the date of this segmentation
            #slice the data to capture this segmentation
            forecast_sliced = forecast[index]
            observed_rain_i = observed_rain[index]

            #add data from this segmentation
            for i_error, error_class in enumerate(ERROR_CLASSES):
                error_all_array[i_error].add_data(
                    forecast_sliced, observed_rain_i)
                error_plot_array[i_error].append(
                    forecast_sliced.get_error(observed_rain_i, error_class()))

        #plot for each error
        pandas.plotting.register_matplotlib_converters()
        for i_error, error_class in enumerate(ERROR_CLASSES):
            plt.figure()
            plt.plot(time_array, error_plot_array[i_error], '-o')
            plt.hlines(error_all_array[i_error].get_error(),
                       time_array[0],
                       time_array[-1],
                       linestyles='dashed')
            plt.xlabel("date")
            plt.ylabel(error_class.get_axis_label())
            plt.savefig(
                path.join(directory,
                          prefix + "_" + error_class.get_short_name() + ".pdf"))
            plt.close()