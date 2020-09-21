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

                bias_loss_plot = []
                for loss_i in self.loss_segment_array[i_loss]:
                    bias_loss_plot.append(loss_i.get_bias_loss())

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

class Downscale(TimeSeries):

    def evaluate_loss(self, forecast, time_segmentator):
        #the forecaster object for downscale already has the test set
        super().evaluate_loss(forecast, None, time_segmentator)

    def evaluate_loss_segment(self, forecast, observed_rain, index):
        #observed_rain unused
        #add data from this segmentation
        for i_loss, Loss in enumerate(LOSS_CLASSES):
            forecast.add_data_to_loss(self.loss_all_array[i_loss], index)
            loss_i = Loss(forecast.n_simulation)
            forecast.add_data_to_loss(loss_i, index)
            self.loss_segment_array[i_loss].append(loss_i)
