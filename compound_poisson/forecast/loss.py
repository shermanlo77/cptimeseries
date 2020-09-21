import math

import numpy as np

class Loss(object):
    """For evaluating the prediction performance when comparing the forecast
        with the observed precipitation

    Use add_data() to add a single time_series (ie for one location) at a time.
        This updates the losses in loss_array. Afterwards, use get_bias_loss(),
        get_risk() or get_loss_quantile() to get statistics of the losses.

    How to implement:
        Implement the methods get_bias_loss(), get_risk() get_loss_quantile(),
            loss_function()
        Implement the functions get_short_name(), get_short_bias_name(),
            get_axis_bias_label(), get_name()

    Attributes:
        loss_array: array of losses, one for each forecast sample/simulation
        bias_loss_sum: the sum of bias loss
        n_time_pointers: number of time points considered after one or more
            calls to add_data()
    """

    def __init__(self, n_simulation):
        self.loss_array = np.zeros(n_simulation)
        self.bias_loss_sum = 0
        self.n_time_points = 0

    def add_data(self, forecast, observed_data):
        """Update member variables (eg losses) with new data

        Args:
            forecast: forecaster object
            observed_data: numpy array of observed precipitation, same time
                length as forecast
        """
        self.n_time_points += len(observed_data)
        self.bias_loss_sum += self.loss_function(
            forecast.forecast_median, observed_data)
        for i_simulation in range(len(self.loss_array)):
            self.loss_array[i_simulation] += self.loss_function(
                forecast.forecast_array[i_simulation], observed_data)

    def get_bias_loss(self):
        """Return a SINGLE forecast (eg median over samples) with the obserbed
            metric, eg root mean bias squared
        """
        raise NotImplementedError

    def get_risk(self):
        """Return the posterior expectation of the loss, aka the risk
        """
        raise NotImplementedError

    def get_loss_quantile(self, quantile):
        """Return the quantiles of the posterior loss
        """
        raise NotImplementedError

    def loss_function(self, prediction, observe):
        """Return the sum of losses
        """
        raise NotImplementedError

    def get_short_name():
        raise NotImplementedError

    def get_short_bias_name():
        raise NotImplementedError

    def get_axis_label():
        raise NotImplementedError

    def get_axis_bias_label():
        raise NotImplementedError

class RootMeanSquareError(Loss):

    def __init__(self,n_simulation):
        super().__init__(n_simulation)

    def loss_function(self, prediction, observe):
        return np.sum(np.square(prediction - observe))

    def get_bias_loss(self):
        return math.sqrt(self.bias_loss_sum / self.n_time_points)

    def get_risk(self):
        return math.sqrt(np.mean(self.loss_array) / self.n_time_points)

    def get_loss_quantile(self, quantile):
        return np.sqrt(
            np.quantile(self.loss_array, quantile) / self.n_time_points)

    def get_short_name():
        return "rmse"

    def get_short_bias_name():
        return "rmsb"

    def get_axis_label():
        return "root mean square error (mm)"

    def get_axis_bias_label():
        return "root mean square bias (mm)"

class RootMeanSquare10Error(RootMeanSquareError):

    def __init__(self, n_simulation):
        super().__init__(n_simulation)

    def add_data(self, forecast, observed_data):
        #override
        is_above_10 = observed_data >= 10
        if np.any(is_above_10):
            observed_10 = observed_data[is_above_10]
            self.n_time_points += len(observed_10)
            self.bias_loss_sum += self.loss_function(
                forecast.forecast_median[is_above_10], observed_10)
            for i_simulation in range(len(self.loss_array)):
                self.loss_array[i_simulation] += self.loss_function(
                    forecast.forecast_array[i_simulation][is_above_10],
                    observed_10)

    def get_bias_loss(self):
        #handle sistuations where it never rained more than 10 mm
        if self.n_time_points == 0:
            return math.nan
        else:
            return super().get_bias_loss()

    def get_risk(self):
        if self.n_time_points == 0:
            return math.nan
        else:
            return math.sqrt(np.mean(self.loss_array) / self.n_time_points)

    def get_loss_quantile(self, quantile):
        if self.n_time_points == 0:
            return math.nan
        else:
            return np.sqrt(
                np.quantile(self.loss_array, quantile) / self.n_time_points)

    def get_short_name():
        return "r10"

    def get_short_bias_name():
        return "rb10"

    def get_axis_label():
        return "root mean square error 10 (mm)"

    def get_axis_bias_label():
        return "root mean square bias 10 (mm)"

class MeanAbsoluteError(Loss):

    def __init__(self, n_simulation):
        super().__init__(n_simulation)

    def loss_function(self, prediction, observe):
        return np.sum(np.absolute(prediction - observe))

    def get_bias_loss(self):
        return self.bias_loss_sum / self.n_time_points

    def get_risk(self):
        return np.mean(self.loss_array) / self.n_time_points

    def get_loss_quantile(self, quantile):
        return np.quantile(self.loss_array, quantile) / self.n_time_points

    def get_short_name():
        return "mae"

    def get_short_bias_name():
        return "mab"

    def get_axis_label():
        return "mean absolute error (mm)"

    def get_axis_bias_label():
        return "mean absolute bias (mm)"
