"""For plotting errors for every segmentation (eg, every year)

Plot the errors for each segmentation
Also plot (as a horizontal line) the error for all segmentations combined
"""

import math
from os import path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas.plotting
from scipy import stats

from compound_poisson.forecast import loss

#list of all the errors to plot
LOSS_CLASSES = [
    loss.RootMeanSquareError,
    loss.RootMeanSquare10Error,
    loss.MeanAbsoluteError,
]

class TimeSeries(object):
    """
    Attributes:
        time_array: array of dates for each segmentation
        loss_all_array: array of loss objects when combining the segmentations
        loss_segment_array: array of arrays, for each loss, each containing
            array of loss objects for each segmentation
    """

    def __init__(self):
        self.time_array = None
        self.loss_all_array = None
        self.loss_segment_array = None

    def evaluate_loss(self,
                      forecast,
                      observed_rain,
                      time_segmentator):
        """
        Args:
            forecast: Forecaster object
            observed_rain: numpy array of observed precipitation
            time_segmentator: TimeSegmenator object
        """
        self.time_array = []
        self.loss_all_array = []
        self.loss_segment_array = []
        #init loss objects and variables
        for Loss in LOSS_CLASSES:
            self.loss_all_array.append(Loss(forecast.n_simulation))
            self.loss_segment_array.append([])
        #for each segmentation
        for date, index in time_segmentator:
            self.time_array.append(date) #get the date of this segmentation
            self.evaluate_loss_segment(forecast, observed_rain, index)

    def evaluate_loss_segment(self, forecast, observed_rain, index):
        #slice the data to capture this segmentation
        forecast_sliced = forecast[index]
        observed_rain_i = observed_rain[index]
        #add data from this segmentation
        for i_error, Loss in enumerate(LOSS_CLASSES):
            self.loss_all_array[i_error].add_data(
                forecast_sliced, observed_rain_i)
            loss_i = Loss(forecast_sliced.n_simulation)
            loss_i.add_data(forecast_sliced, observed_rain_i)
            self.loss_segment_array[i_error].append(loss_i)

    def plot_loss(self, directory, prefix=""):
        #it is possible for the time_array to be empty, for example, r10 would
            #be empty is it never rained more than 10 mm
        if self.time_array:
            #plot for each loss
            pandas.plotting.register_matplotlib_converters()
            for i_loss, Loss in enumerate(LOSS_CLASSES):

                #bias loss for each segment
                bias_loss_plot = []
                for loss_i in self.loss_segment_array[i_loss]:
                    bias_loss_plot.append(loss_i.get_bias_loss())

                #posterior loss quantiles
                loss_lower_array = []
                loss_array = []
                loss_upper_array = []
                quantile_array = stats.norm.cdf([-1, 0, 1])
                for loss_i in self.loss_segment_array[i_loss]:
                    loss_quantile = loss_i.get_loss_quantile(quantile_array)
                    #it is possible for loss_quantile to be nan, this happens
                        #if there is a time segment with no occurance of 10 mm >
                        #of rain for r10 lost
                    if np.any(np.isnan(loss_quantile)):
                        loss_quantile = np.full_like(quantile_array, math.nan)
                    loss_lower_array.append(loss_quantile[0])
                    loss_array.append(loss_quantile[1])
                    loss_upper_array.append(loss_quantile[2])

                #plot the bias loss
                plt.figure()
                plt.plot(self.time_array, bias_loss_plot, '-o')
                plt.hlines(self.loss_all_array[i_loss].get_bias_loss(),
                           self.time_array[0],
                           self.time_array[-1],
                           linestyles='dashed')
                plt.xlabel("date")
                plt.ylabel(Loss.get_axis_bias_label())
                plt.savefig(
                    path.join(directory,
                              (prefix + "_" + Loss.get_short_bias_name()
                                  + ".pdf")))
                plt.close()

                #plot the posterior loss with credible interval
                colours = matplotlib.rcParams['axes.prop_cycle'].by_key()
                colours = colours['color']
                plt.figure()
                plt.fill_between(self.time_array,
                                 loss_lower_array,
                                 loss_upper_array,
                                 alpha=0.25,
                                 color=colours[0])
                plt.plot(self.time_array, loss_array, '-o', color=colours[0])
                plt.hlines(self.loss_all_array[i_loss].get_loss_quantile(0.5),
                           self.time_array[0],
                           self.time_array[-1],
                           linestyles='dashed')
                plt.xlabel("date")
                plt.ylabel(Loss.get_axis_label())
                plt.savefig(
                    path.join(directory,
                              (prefix + "_" + Loss.get_short_name()
                                  + ".pdf")))
                plt.close()

class Downscale(TimeSeries):

    def evaluate_loss(self, forecast, time_segmentator):
        #the forecaster object for downscale already has the test set
        super().evaluate_loss(forecast, None, time_segmentator)

    def evaluate_loss_segment(self, forecast, observed_rain, index):
        #observed_rain unused
        #add data from this segmentation
        for i_loss, Loss in enumerate(LOSS_CLASSES):
            self.loss_all_array[i_loss].add_downscale_forecaster(
                forecast, index)
            loss_i = Loss(forecast.n_simulation)
            loss_i.add_downscale_forecaster(forecast, index)
            self.loss_segment_array[i_loss].append(loss_i)
