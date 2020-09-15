import math

import numpy as np

class Error(object):
    """For evaluating the prediction performance when comparing the forecast
        with the observed precipitation

    Use add_data() to add a single time_series (ie for one location) at a time.
        Afterwards, use get_error to evaluate the error().
    """

    def __init__(self):
        pass

    def add_data(self, forecast, observed_data):
        raise NotImplementedError

    def get_error(self):
        raise NotImplementedError

    def get_short_name():
        raise NotImplementedError

    def get_name():
        raise NotImplementedError

class RootMeanSquareError(Error):

    def __init__(self):
        self.n = 0;
        self.sum_square = 0;

    def add_data(self, forecast, observed_data):
        self.n += len(observed_data)
        self.sum_square += np.sum(
            np.square(forecast.forecast_median - observed_data))

    def get_error(self):
        return math.sqrt(self.sum_square / self.n)

    def get_short_name():
        return "rmse"

    def get_axis_label():
        return "root mean square error"

class RootMeanSquare10Error(RootMeanSquareError):

    def __init__(self):
        super().__init__()

    def add_data(self, forecast, observed_data):
        is_above_10 = observed_data >= 10
        observed_10 = observed_data[is_above_10]
        forecast_10 = forecast.forecast_median[is_above_10]
        self.n += len(observed_10)
        self.sum_square += np.sum(np.square(forecast_10 - observed_10))

    def get_short_name():
        return "r10"

    def get_axis_label():
        return "rmse 10"

class MeanAbsoluteError(Error):

    def __init__(self):
        super().__init__()
        self.n = 0;
        self.sum_error = 0;

    def add_data(self, forecast, observed_data):
        self.n += len(observed_data)
        self.sum_error += np.sum(
            np.absolute(forecast.forecast_median - observed_data))

    def get_error(self):
        return self.sum_error / self.n

    def get_short_name():
        return "mae"

    def get_axis_label():
        return "mean absolute error"