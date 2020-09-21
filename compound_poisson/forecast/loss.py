import math

import numpy as np

class Loss(object):
    """For evaluating the prediction performance when comparing the forecast
        with the observed precipitation

    Use add_data() to add a single time_series (ie for one location) at a time.
        Afterwards, use get_error to evaluate the error().
    """

    def __init__(self, n_simulation):
        self.loss_array = np.zeros(n_simulation)
        self.bias_squared_sum = 0
        self.n_time_points = 0

    def add_data(self, forecast, observed_data):
        self.n_time_points += len(observed_data)
        self.bias_squared_sum += self.loss_function(
            forecast.forecast_median, observed_data)
        for i_simulation in range(len(self.loss_array)):
            self.loss_array += self.loss_function(
                forecast.forecast_array[i_simulation], observed_data)

    def get_root_bias_squared(self):
        raise NotImplementedError

    def loss_function(self, prediction, observe):
        """Return the sum of losses
        """
        raise NotImplementedError

    def get_short_name():
        raise NotImplementedError

    def get_name():
        raise NotImplementedError

class RootMeanSquareError(Loss):

    def __init__(self,n_simulation):
        super().__init__(n_simulation)

    def loss_function(self, prediction, observe):
        return np.sum(np.square(prediction - observe))

    def get_root_bias_squared(self):
        return math.sqrt(self.bias_squared_sum / self.n_time_points)

    def get_short_name():
        return "rmse"

    def get_axis_label():
        return "root mean square error (mm)"

class RootMeanSquare10Error(RootMeanSquareError):

    def __init__(self, n_simulation):
        super().__init__(n_simulation)

    def add_data(self, forecast, observed_data):
        is_above_10 = observed_data >= 10
        if np.any(is_above_10):
            observed_10 = observed_data[is_above_10]
            self.n_time_points += len(observed_10)
            self.bias_squared_sum += self.loss_function(
                forecast.forecast_median[is_above_10], observed_10)
            for i_simulation in range(len(self.loss_array)):
                self.loss_array += self.loss_function(
                    forecast.forecast_array[i_simulation][is_above_10],
                    observed_10)

    def get_root_bias_squared(self):
        #handle sistuations where it never rained more than 10 mm
        if self.n_time_points == 0:
            return math.nan
        else:
            return super().get_root_bias_squared()

    def get_short_name():
        return "r10"

    def get_axis_label():
        return "rmse 10 (mm)"

class MeanAbsoluteError(Loss):

    def __init__(self, n_simulation):
        super().__init__(n_simulation)

    def loss_function(self, prediction, observe):
        return np.sum(np.absolute(prediction - observe))

    def get_root_bias_squared(self):
        return self.bias_squared_sum / self.n_time_points

    def get_short_name():
        return "mae"

    def get_axis_label():
        return "mean absolute error (mm)"
