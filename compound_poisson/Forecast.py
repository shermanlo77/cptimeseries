import math
import matplotlib.pyplot as plot
import numpy as np
import scipy.stats as stats

class Forecast:
    """Contain Monte Carlo forecasts
    
    Contain Monte Carlo forecasts in forecast_array as an array of TimeSeries
        objects. Add forecasts using the method append(). Call the method
        get_forecast() to calculate statistcs of all forecats. The statistics
        are stored as member variables.
    Used by the methods TimeSeries.forecast() and TimeSeries.forecast_self().
    
    Attributes:
        forecast_array: array of TimeSeries objects
        forecast_numpy: forecast_array in numpy format
        forecast: expectation of all forecasts
        forecast_error: std of all forecasts
        forecast_median: median of all forecasts
        forecast_sigma: dictoary of z sigma errors of all forecasts
        time_array: array, containing time stamps for each point in the forecast
        n: length of time series
        n_simulation: length of forecast_array
    """
    
    def __init__(self):
        self.forecast_array = []
        self.forecast_numpy = None
        self.forecast = None
        self.forecast_error = None
        self.forecast_median = None
        self.forecast_sigma = {}
        self.time_array = None
        self.n = None
        self.n_simulation = 0
    
    def append(self, forecast):
        """Append forecast to collection
        """
        self.forecast_array.append(forecast)
        self.n_simulation += 1
        self.n = len(forecast)
    
    def get_forecast(self):
        """Calculate statistics over all the provided forecasts
        """
        forecast_array = []
        for forecast in self.forecast_array:
            forecast_array.append(forecast.y_array)
        self.forecast_numpy = np.asarray(forecast_array)
        self.forecast = np.mean(forecast_array, 0)
        self.forecast_error = np.std(forecast_array, 0, ddof=1)
        sigma_array = range(-3,4)
        forecast_quantile = np.quantile(forecast_array,
                                        stats.norm.cdf(sigma_array),
                                        0)
        for i in range(len(sigma_array)):
            self.forecast_sigma[sigma_array[i]] = forecast_quantile[i]
        self.forecast_median = self.forecast_sigma[0]
        self.time_array = self.forecast_array[0].time_array
    
    def get_prob_rain(self, rainfall):
        """Get the probability if it will rain at least of a certian amount
        
        Args:
            rainfall: scalar, amount of rain to evaluate the probability
        
        Return:
            vector, a probability for each day
        """
        p_rain = np.sum(self.forecast_numpy > rainfall, 0) / self.n_simulation
        return p_rain
    
    def plot_roc_curve(self, rainfall, true_y):
        """Plot ROC curve
        """
        #thresholds where a marker is shown
        marker_threshold = [0.05, 0.10, 0.20]
        #markers for each of these thresholds
        marker_array = ['o', '^', 's']
        #get probability it will rain more than rainfall
        p_rain = self.get_prob_rain(rainfall)
        #for each positive probability, sort them and they will be used for
            #thresholds
        threshold_array = np.sort(p_rain)
        threshold_array = threshold_array[threshold_array > 0]
        #marker thresholds goes at the end of threshold_array
        threshold_array = np.concatenate((threshold_array, marker_threshold))
        
        #get the times it rained more than rainfall
        is_rain = true_y > rainfall
        n_is_rain = np.sum(is_rain) #number of times the event happened
        #number of times event did not happaned
        n_is_not_rain = len(is_rain) - n_is_rain
        #array to store true and false positives, used for plotting
        true_positive_array = []
        false_positive_array = []
        #for each threshold, get true and false positive
        for threshold in threshold_array:
            positive = p_rain >= threshold
            true_positive = (
                np.sum(np.logical_and(positive, is_rain)) / n_is_rain)
            false_positive = (
                np.sum(np.logical_and(positive, np.logical_not(is_rain)))
                / n_is_not_rain)
            true_positive_array.append(true_positive)
            false_positive_array.append(false_positive)
        
        #variables for extracting ordered thresholds and marker thresholds
        n_marker = len(marker_threshold)
        n_threshold = len(threshold_array) - n_marker
        #plot ROC curve, append [1] and [0] so that curves start at (1,1) and
            #ends at (0,0)
        #thresholds goes from smallest to largest, i.e highest false positive
            #rate to lowest false positive rate 
        ax = plot.plot(np.concatenate(
                      ([1], false_positive_array[0:n_threshold], [0])),
                  np.concatenate(
                      ([1], true_positive_array[0:n_threshold], [0])),
                  label=str(rainfall)+" mm",
                  )
        colour = ax[0].get_color() #make scatter plot same colour as ROC curve
        #scatter plot the marker thresholds
        for i in range(len(marker_threshold)):
            plot.scatter(false_positive_array[n_threshold + i],
                         true_positive_array[n_threshold + i],
                         marker=marker_array[i],
                         c=colour)
        plot.xlabel("false positive rate")
        plot.ylabel("true positive rate")
    
    def get_error_rmse(self, true_y):
        """Return root mean square error
        
        Compare forecast with a given true_y using the root mean squared error
        
        Args:
            true_y: array of y
        
        Return:
            root mean square error
        """
        n = len(true_y)
        return np.sqrt(np.sum(np.square(self.forecast - true_y)) / n)
    
    def get_error_square_sqrt(self, true_y):
        """Return square mean sqrt error
        
        Compare forecast with a given true_y using the square mean sqrt error.
            This is the Tweedie deviance with p = 1.5
        
        Args:
            true_y: array of y
        
        Return:
            square mean sqrt error, can be infinite
        """
        n = len(true_y)
        error = np.zeros(n)
        is_infinite = False
        for i in range(n):
            y = true_y[i]
            y_hat = self.forecast[i]
            if y == 0:
                error[i] = math.sqrt(y_hat)
            else:
                if y_hat == 0:
                    is_infinite = True
                    break
                else:
                    sqrt_y_hat = math.sqrt(y_hat)
                    error[i] = (4 * math.pow(math.sqrt(y) - sqrt_y_hat, 2)
                        / sqrt_y_hat)
        if is_infinite:
            return math.inf
        else:
            return math.pow(np.sum(error) / n, 2)
