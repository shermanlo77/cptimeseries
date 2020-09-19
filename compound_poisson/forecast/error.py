import math

import numpy as np

class Error(object):
    """For evaluating the prediction performance when comparing the forecast
        with the observed precipitation

    Use add_data() to add a single time_series (ie for one location) at a time.
        Afterwards, use get_error to evaluate the error().
    """

    def __init__(self):
        self.error_array = []
        self.bias_squared_sum = 0

    def add_data(self, forecast, observed_data):
        raise NotImplementedError

    def get_root_bias_squared(self):
        raise NotImplementedError

    def get_short_name():
        raise NotImplementedError

    def get_name():
        raise NotImplementedError

class RootMeanSquareError(Error):

    def __init__(self):
        super().__init__()
        self.n = 0

    def add_data(self, forecast, observed_data):
        self.bias_squared_sum += self.square_error(
            forecast.forecast_median, observed_data)
        self.n += len(observed_data)

    def square_error(self, prediction, observe):
        return np.sum(np.square(prediction - observe))

    def get_root_bias_squared(self):
        return math.sqrt(self.bias_squared_sum / self.n)

    def get_short_name():
        return "rmse"

    def get_axis_label():
        return "root mean square error (mm)"

class RootMeanSquare10Error(RootMeanSquareError):

    def __init__(self):
        super().__init__()

    def add_data(self, forecast, observed_data):
        is_above_10 = observed_data >= 10
        if np.any(is_above_10):
            observed_10 = observed_data[is_above_10]
            forecast_10 = forecast.forecast_median[is_above_10]
            self.n += len(observed_10)
            self.bias_squared_sum += self.square_error(forecast_10, observed_10)

    def get_root_bias_squared(self):
        #handle sistuations where it never rained more than 10 mm
        if self.n == 0:
            return math.nan
        else:
            return super().get_root_bias_squared()

    def get_short_name():
        return "r10"

    def get_axis_label():
        return "rmse 10 (mm)"

class MeanAbsoluteError(Error):

    def __init__(self):
        super().__init__()
        self.n = 0;

    def add_data(self, forecast, observed_data):
        self.n += len(observed_data)
        self.bias_squared_sum += self.absolute_error(
            forecast.forecast_median, observed_data)

    def absolute_error(self, prediction, observe):
        return np.sum(np.absolute(prediction - observe))

    def get_root_bias_squared(self):
        return self.bias_squared_sum / self.n

    def get_short_name():
        return "mae"

    def get_axis_label():
        return "mean absolute error (mm)"
