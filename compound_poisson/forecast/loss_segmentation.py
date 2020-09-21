"""For plotting errors for every segmentation (eg, every year)

Plot the errors for each segmentation
Also plot (as a horizontal line) the error for all segmentations combined
"""

from os import path

import matplotlib.pyplot as plt
import pandas.plotting

from compound_poisson.forecast import loss

#list of all the errors to plot
LOSS_CLASSES = [
    loss.RootMeanSquareError,
    loss.RootMeanSquare10Error,
    loss.MeanAbsoluteError,
]

class TimeSeries(object):

    def __init__(self):
        pass

    def plot_loss(self,
                  forecast,
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
        #array of loss objects when combining the segmentations
        loss_all_array = []
        #array of arrays, for each loss, each containing loss for each
            #segmentation
        loss_plot_array = []

        #init loss objects and variables
        for Loss in LOSS_CLASSES:
            loss_all_array.append(Loss())
            loss_plot_array.append([])

        #for each segmentation
        for date, index in time_segmentator:
            time_array.append(date) #get the date of this segmentation

            self.evaluate_loss(forecast,
                               observed_rain,
                               index,
                               loss_all_array,
                               loss_plot_array)

        #it is possible for the time_array to be empty, for example, r10 would
            #be empty is it never rained more than 10 mm
        if time_array:
            #plot for each loss
            pandas.plotting.register_matplotlib_converters()
            for i_loss, loss_class in enumerate(LOSS_CLASSES):
                plt.figure()
                plt.plot(time_array, loss_plot_array[i_loss], '-o')
                plt.hlines(loss_all_array[i_loss].get_root_bias_squared(),
                           time_array[0],
                           time_array[-1],
                           linestyles='dashed')
                plt.xlabel("date")
                plt.ylabel(loss_class.get_axis_label())
                plt.savefig(
                    path.join(directory,
                              (prefix + "_" + loss_class.get_short_name()
                                  + ".pdf")))
                plt.close()

    def evaluate_loss(self,
                       forecast,
                       observed_rain,
                       index,
                       loss_all_array,
                       loss_plot_array):
        #slice the data to capture this segmentation
        forecast_sliced = forecast[index]
        observed_rain_i = observed_rain[index]
        #add data from this segmentation
        for i_error, loss_class in enumerate(LOSS_CLASSES):
            loss_all_array[i_error].add_data(
                forecast_sliced, observed_rain_i)
            loss_plot_array[i_error].append(
                forecast_sliced.get_error(observed_rain_i, loss_class()))

class Downscale(TimeSeries):

    def plot_loss(self,
                  forecast,
                  time_segmentator,
                  directory,
                  prefix=""):
        #the forecaster object for downscale already has the test set
        super().plot_loss(forecast, None, time_segmentator, directory, prefix)

    def evaluate_loss(self,
                      forecast,
                      observed_rain,
                      index,
                      loss_all_array,
                      loss_plot_array):
        #observed_rain unused
        #add data from this segmentation
        for i_loss, loss_class in enumerate(LOSS_CLASSES):
            forecast.add_data_to_loss(loss_all_array[i_loss], index)
            loss_plot_array[i_loss].append(
                forecast.get_error(loss_class(), index))
