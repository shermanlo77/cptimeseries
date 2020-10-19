"""For evaluating and plotting losses for every segmentation (eg, every year)

Designed to handle the different Loss, mean/median bias and different
    TimeSegmenator.

How to use:
    -Pass the forecasted and observed data via the constructor
    -Call the method evaluate_loss() to evaluate the loss for a given
        TimeSegmenator. This can be called multiple times for different
        time_segmentator, with results resetting for each call.
    -Extract Loss objects from the member variables.
    -Call the method plot_loss() to plot the loss for each time segment.

TimeSeries <- Downscale
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
    loss.MeanAbsolute10Error,
]

class TimeSeries(object):
    """
    Attributes:
        forecaster: forecast.time_series.Forecaster object
        observed_rain: numpy array of observed rain
        time_array: array of dates for each segmentation
        loss_all_array: array of loss objects when combining the segmentations,
            each element for each Loss in LOSS_CLASSES
        loss_segment_array: array of arrays of Loss objects
            dim 0: for each loss in LOSS_CLASSES
            dim 1: for each segmentation
    """

    def __init__(self, forecaster, observed_rain):
        """
        Args:
            forecaster: forecast.time_series.Forecaster object
            observed_rain: numpy array of observed rain
        """
        self.forecaster = forecaster
        self.observed_rain = observed_rain
        self.time_array = None
        self.loss_all_array = None
        self.loss_segment_array = None

    def evaluate_loss(self, time_segmentator):
        """Evaluate the loss for a given time_segmentator and update the member
            variables.

        Can be called multiple times with a different time_segmentator. Results
            are reset between each call.

        Args:
            time_segmentator: TimeSegmenator object
        """
        self.time_array = []
        self.loss_all_array = []
        self.loss_segment_array = []
        #init loss objects and variables
        for Loss in LOSS_CLASSES:
            self.loss_all_array.append(Loss(self.forecaster.n_simulation))
            self.loss_segment_array.append([])
        #for each segmentation
        for date, index in time_segmentator:
            self.time_array.append(date) #get the date of this segmentation
            self.evaluate_loss_segment(index)

    def evaluate_loss_segment(self, index):
        """For a given segment, instantiate a new Loss object and add data to
            it. Also add data to the losses in self.loss_all_array. Member
            variables are updated.

        Args:
            index: slice object pointing to a time segment
        """
        #slice the data to capture this segmentation
        forecaster_slice = self.forecaster[index]
        observed_rain_slice = self.observed_rain[index]
        #add data from this segmentation for each loss
        for i_error, Loss in enumerate(LOSS_CLASSES):
            #add data to the loss objects which cover all time segments
            self.loss_all_array[i_error].add_data(
                forecaster_slice, observed_rain_slice)
            #new loss for this segment
            loss_i = Loss(forecaster_slice.n_simulation)
            loss_i.add_data(forecaster_slice, observed_rain_slice)
            self.loss_segment_array[i_error].append(loss_i)

    def plot_loss(self, directory, prefix="", cycler=None):
        """Plot the errors for each segmentation and (as a
            horizontal line) the error for all segmentations combined. All
            losses and expectation bias and median bias are considered. Figures
            are saved.
        """
        #it is possible for the time_array to be empty, for example, r10 would
            #be empty is it never rained more than 10 mm
        if self.time_array:
            #plot for each loss
            pandas.plotting.register_matplotlib_converters()
            for i_loss, Loss in enumerate(LOSS_CLASSES):

                bias_loss_plot, bias_median_loss_plot = self.get_bias_plot(
                    i_loss)

                #bias of the mean
                self.plot(bias_loss_plot,
                          self.loss_all_array[i_loss].get_bias_loss(),
                          Loss.get_axis_bias_label(),
                          path.join(directory,
                                    (prefix + Loss.get_short_bias_name()
                                        + "_mean.pdf")),
                          cycler)

                #bias of the median
                self.plot(bias_median_loss_plot,
                          self.loss_all_array[i_loss].get_bias_median_loss(),
                          Loss.get_axis_bias_label(),
                          path.join(directory,
                                    (prefix + Loss.get_short_bias_name()
                                        + "_median.pdf")),
                          cycler)

    def get_bias_plot(self, i_loss):
        """Return array of values of bias loss for each time segment

        Args:
            i_loss: integer, pointing to an element in LOSS_CLASSES

        Return:
            bias_loss_plot: array of bias loss for each time segment
            bias_median_loss_plot: array of bias loss (using the median) for
                each time segment.
        """
        #bias loss for each segment
        bias_loss_plot = []
        bias_median_loss_plot = []
        for loss_i in self.loss_segment_array[i_loss]:
            bias_loss_plot.append(loss_i.get_bias_loss())
            bias_median_loss_plot.append(loss_i.get_bias_median_loss())
        return (bias_loss_plot, bias_median_loss_plot)

    def plot(self, plot_array, h_line, label_axis, path_to_fig, cycler=None):
        """A basic plot method

        Args:
            plot_array: array of values to plot for each time in self.time_array
            h_line: horizontal line to plot
            label_axis: parameter for plt.ylabel
            path_to_fig: where to save the figure:
            cycler: optional cycler to use when plotting
        """
        plt.figure()
        if not cycler is None:
            ax = plt.gca()
            ax.set_prop_cycle(cycler)
        plt.plot(self.time_array, plot_array)
        plt.hlines(h_line,
                   self.time_array[0],
                   self.time_array[-1],
                   linestyles='dashed')
        plt.xlabel("date")
        plt.xticks(rotation=45)
        plt.ylabel(label_axis)
        plt.savefig(path_to_fig, bbox_inches="tight")
        plt.close()

class Downscale(TimeSeries):
    """
    Attributes:
        forecaster: forecast.downscale.Forecaster object
        observed_rain: NOT USED
    All remaining attributes are as superclass.
    """

    def __init__(self, forecaster):
        """
        Args:
            forecaster: forecast.downscale.Forecaster object
        """
        #test set already lives in forecaster
        super().__init__(forecaster, None)

    #override
    def evaluate_loss_segment(self, index):
        #observed_rain unused
        #add data for this segmentation
        for i_loss, Loss in enumerate(LOSS_CLASSES):
            self.loss_all_array[i_loss].add_downscale_forecaster(
                self.forecaster, index)
            loss_i = Loss(self.forecaster.n_simulation)
            loss_i.add_downscale_forecaster(self.forecaster, index)
            self.loss_segment_array[i_loss].append(loss_i)
