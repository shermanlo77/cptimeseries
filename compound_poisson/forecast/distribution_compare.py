"""For comparing the distribution of the forecast with the observed.

The distribution of the forecast is obtained by evaluating get_prob_rain(x) for
    each time point and taking the average over time. This is the same with the
    observed (each time point would either give 0.0 or 1.0). This would work for
    ERA5 as well.
There are methods for plotting the survival functions, pp plots and qq plots.

How to use:
    -Instantiate either a TimeSeries or Downscale object
    -Call the method compare(), this would modify the member variables
    -Call methods to plot results

TimeSeries <- Downscale
"""

import math

from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

class TimeSeries(object):
    """
    Attributes:
        observed_array: array of precipitation the survival functions were
            evaluated (x-axis for survival function)
        prob_forecast_array: array of survival functions evaluated at
            observed_array for the forecast (y-axis for survival function)
        prob_observed_array: array of survival functions evaluated at
            observed_array for the observed (y-axis for survival function)
        qq_observe: array of observed precipitation (same quantiles
            element-wise with qq_forecast) (x-axis for qq plot)
        qq_forecast: array of forecasted precipitation (same quantiles
            element-wise with qq_observe) (y-axis for qq plot)
    """

    def __init__(self):
        self.observed_array = None
        self.prob_forecast_array = []
        self.prob_observed_array = []
        self.qq_observe = None
        self.qq_forecast = None

    def compare(self, forecaster, observed_rain, n_linspace):
        """
        Args:
            forecaster: compound_poisson.forecast.time_series.Forecaster object
            observed_rain: numpy array of observed precipitation
            n_linspace: number of points to evaluate between 0 mm and max
                observed rain
        """
        #range of observed precipitation to plot in the qq plot
        self.observed_array = np.linspace(0, observed_rain.max(), n_linspace)

        #get prob(precipitation > rain) for each rain in observed_array
        self.prob_forecast_array = []
        self.prob_observed_array = []
        for rain in self.observed_array:
            self.prob_forecast_array.append(
                np.mean(forecaster.get_prob_rain(rain)))
            self.prob_observed_array.append(np.mean(observed_rain > rain))

        self.get_qq_plot()

    def plot_survival(self):
        self.plot_survival_forecast("forecast")
        self.plot_survival_observed("observed")

    def plot_survival_forecast(self, label=None):
        plt.plot(self.observed_array, self.prob_forecast_array, label=label)

    def plot_survival_observed(self, label=None):
        plt.plot(self.observed_array, self.prob_observed_array, label=label)

    def adjust_survival_plot(self):
        plt.xlim([0,self.observed_array[len(self.qq_observe)-1]])
        plt.xlabel("x (mm)")
        plt.ylabel("probability of precipitation > x")

    def plot_pp(self, label=None):
        cdf_observed = 1 - np.asarray(self.prob_observed_array)
        cdf_forecast = 1 - np.asarray(self.prob_forecast_array)

        plt.scatter(cdf_observed[0], cdf_forecast[0])
        plt.plot(cdf_observed, cdf_forecast, label=label)

    def adjust_pp_plot(self):
        plt.plot([0, 1], [0, 1], 'k:')
        plt.xlabel("distribution of observed precipitation")
        plt.ylabel("distribution of forecasted precipitation")

    def plot_qq(self, label=None):
        plt.plot(self.qq_observe, self.qq_forecast, label=label)

    def adjust_qq_plot(self):
        ax = plt.gca()
        axis_lim = np.array([ax.get_xlim()[1], ax.get_ylim()[1]])
        axis_lim = axis_lim.min()
        plt.xlim([0, axis_lim])
        plt.ylim([0, axis_lim])
        plt.plot([0, axis_lim], [0, axis_lim], 'k:')
        plt.xlabel("observed precipitation (mm)")
        plt.ylabel("forecasted precipitation (mm)")

    def get_qq_plot(self):
        """Set the member variables qq_forecast and qq_observe, used to plot qq
            plot for comparing distribution of precipitation of the forecast
            with the observed
        """
        #inverse of the probability using interpolation (switch x and y)
        prob_forecast_inverse = interpolate.interp1d(
            self.prob_forecast_array, self.observed_array)

        #for each probability in prob_observed_array, evaluate the inverse of
            #prob_forecast_array using prob_forecast_invers
        #caution when handling 0 mm
        #return nan when ValueError caught, eg when outside the interpolation
            #zone
        self.qq_forecast = []
        for prob_observed in self.prob_observed_array:
            forecast_i = None
            if prob_observed > self.prob_forecast_array[0]:
                forecast_i = 0
            else:
                if prob_observed > 0:
                    try:
                        forecast_i = (
                            prob_forecast_inverse(prob_observed).tolist())
                    except ValueError:
                        forecast_i = math.nan
                else:
                    forecast_i = math.nan
            self.qq_forecast.append(forecast_i)

        self.qq_forecast = np.asarray(self.qq_forecast)

        is_finite = np.isfinite(self.qq_forecast)
        self.qq_forecast = self.qq_forecast[is_finite]
        self.qq_observe = self.observed_array[is_finite]

class Downscale(TimeSeries):

    def __init__(self):
        super().__init__()

    #override
    def compare(self, forecaster, n_linspace):
        """
        Args:
            forecaster: compound_poisson.forecast.downscale.Forecaster object
            n_linspace: number of points to evaluate between 0 mm and max
                observed rain
        """
        self.observed_array = np.linspace(
            0, forecaster.data.rain.max(), n_linspace)

        self.prob_forecast_array = []
        self.prob_observed_array = []
        for rain in self.observed_array:
            self.prob_forecast_array.append(
                np.mean(forecaster.get_prob_rain(rain)))
            self.prob_observed_array.append(
                np.mean(forecaster.data.rain > rain))

        self.get_qq_plot()
