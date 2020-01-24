import math
import numpy as np
import scipy.stats as stats

class Forecast:
    
    def __init__(self):
        self.forecast_array = []
        self.forecast = None
        self.forecast_error = None
        self.forecast_median = None
        self.forecast_sigma = {}
        self.time_array = None
    
    def append(self, forecast):
        self.forecast_array.append(forecast)
    
    def get_forecast(self):
        forecast_array = []
        for forecast in self.forecast_array:
            forecast_array.append(forecast.y_array)
        forecast_array = np.asarray(forecast_array)
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
    
    def get_error_rmse(self, true_y):
        """Return root mean square error
        
        Compare this y_array (usually a prediction) with a given true_y using
            the root mean squared error
        
        Args:
            true_y: array of y
        
        Return:
            root mean square error
        """
        n = len(true_y)
        return np.sqrt(np.sum(np.square(self.forecast - true_y)) / n)
    
    def get_error_square_sqrt(self, true_y):
        """Return square mean sqrt error
        
        Compare this y_array (usually a prediction) with a given true_y using
            the square mean sqrt error. This is the Tweedie deviance with
            p = 1.5
        
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
