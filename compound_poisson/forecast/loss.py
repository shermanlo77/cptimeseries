"""Classes for evaluating the bias loss where the bias is (expectation[forecast]
    - observed) and the loss is some function, eg square. The expectation can be
    replaced with the median.

The 10 means only consider observed precipitation above 10 mm.

Loss <- RootMeanSquareError <- RootMeanSquare10Error
     <- MeanAbsoluteError <- MeanAbsolute10Error
"""

import math

import numpy as np

class Loss(object):
    """For evaluating the prediction performance when comparing the forecast
        with the observed precipitation

    Use add_data() to add a single time_series (ie for one location) at a time.
        This updates the losses in loss_array. Afterwards, use get_bias_loss(),
        get_risk() or get_loss_quantile() to get statistics of the losses.

    NOT IMPLEMENTED:
        loss_array, get_risk() and get_loss_quantile() are not implemented. Only
            the bias is considered here.

    How to implement:
        Implement the methods loss_function(), set_to_orginial_units(),
            get_short_name(), get_short_bias_name(), get_axis_label(),
            get_axis_bias_label()

    Attributes:
        loss_array: array of losses, one for each forecast sample/simulation
        bias_loss_sum: the sum of bias loss
        bias_median_loss_sum: the sum of bias loss where the expectation is
            replaced with the median
        n_time_points: number of time points (x spatial points) considered after
            one or more calls to add_data()
    """

    def __init__(self, n_simulation):
        """
        Args:
            n_simulation: number of predictive posterior samples
        """
        self.loss_array = np.zeros(n_simulation)
        self.bias_loss_sum = 0
        self.bias_median_loss_sum = 0
        self.n_time_points = 0

    def add_data(self, forecast, observed_data, index=None):
        """Update member variables (eg bias_loss_sum) with new data for a single
            location. For multiple locations, see add_downscale_forecaster()

        Args:
            forecast: forecaster.time_series.TimeSeries object
            observed_data: numpy array of observed precipitation, same time
                length as forecast
            index: array of booleans, only consider data with True
        """

        forecast_expectation = None
        forecast_median = None
        #use index to get a subset of the forecast and data
        if index is None:
            forecast_expectation = forecast.forecast
            forecast_median = forecast.forecast_median
        else:
            observed_data = observed_data[index]
            forecast_expectation = forecast.forecast[index]
            forecast_median = forecast.forecast_median[index]

        self.n_time_points += len(observed_data)
        self.bias_loss_sum += self.loss_function(
            forecast_expectation, observed_data)
        self.bias_median_loss_sum += self.loss_function(
            forecast_median, observed_data)

    def add_downscale_forecaster(self, forecaster, index=None):
        """Update member variables (eg losses) with new data for multiple
            locations

        Args:
            forecaster: forecast.downscale.Downscale object
            index: optional, index of times
        """
        for forecaster_i, observed_rain_i in (
            zip(forecaster.generate_forecaster_no_memmap(),
                forecaster.data.generate_unmask_rain())):
            if not index is None:
                forecaster_i = forecaster_i[index]
                observed_rain_i = observed_rain_i[index]
            self.add_data(forecaster_i, observed_rain_i)

    def get_bias_loss(self):
        """Return the mean bias loss
        """
        if self.n_time_points == 0:
            return math.nan
        else:
            return self.set_to_orginial_units(
                self.bias_loss_sum / self.n_time_points)

    def get_bias_median_loss(self):
        """Return the mean bias loss where the bias uses the median
        """
        if self.n_time_points == 0:
            return math.nan
        else:
            return self.set_to_orginial_units(
                self.bias_median_loss_sum / self.n_time_points)

    def get_risk(self):
        """UNUSED: Return the posterior expectation of the loss, aka the risk
        """
        if self.n_time_points == 0:
            return math.nan
        else:
            return self.set_to_orginial_units(
                np.mean(self.loss_array) / self.n_time_points)

    def get_loss_quantile(self, quantile):
        """Return the quantiles of the posterior loss
        """
        if self.n_time_points == 0:
            return math.nan
        else:
            return self.set_to_orginial_units(
                np.quantile(self.loss_array, quantile) / self.n_time_points)

    def loss_function(self, prediction, observe):
        """Return the loss function given a forecast and observed vector
        """
        raise NotImplementedError

    def set_to_orginial_units(self, loss):
        """Normalise the loss to the units of the orginial data
        """
        raise NotImplementedError

    def get_short_name():
        """Name of this loss
        """
        raise NotImplementedError

    def get_short_bias_name():
        """Name of the bias loss
        """
        raise NotImplementedError

    def get_axis_label():
        """Axis label for this loss
        """
        raise NotImplementedError

    def get_axis_bias_label():
        """Axis label for the bias loss
        """
        raise NotImplementedError

class RootMeanSquareError(Loss):

    def __init__(self,n_simulation):
        super().__init__(n_simulation)

    #implemented
    def loss_function(self, prediction, observe):
        """The sum of squared loss
        """
        return np.sum(np.square(prediction - observe))

    #implemented
    def set_to_orginial_units(self, loss):
        #square root (mean of squared loss), aka root mean square
        return np.sqrt(loss)

    #implemented
    def get_short_name():
        return "rmse"

    #implemented
    def get_short_bias_name():
        return "rmsb"

    #implemented
    def get_axis_label():
        return "root mean square error (mm)"

    #implemented
    def get_axis_bias_label():
        return "root mean square bias (mm)"

class RootMeanSquare10Error(RootMeanSquareError):

    def __init__(self, n_simulation):
        super().__init__(n_simulation)

    #override
    def add_data(self, forecast, observed_data):
        #only consider observed data with precipitation greater than 10 mm
        is_above_10 = observed_data >= 10
        if np.any(is_above_10):
            super().add_data(forecast, observed_data, is_above_10)

    #override
    def get_short_name():
        return "rmse10"

    #override
    def get_short_bias_name():
        return "rmsb10"

    #override
    def get_axis_label():
        return "root mean square error 10 (mm)"

    #override
    def get_axis_bias_label():
        return "root mean square bias 10 (mm)"

class MeanAbsoluteError(Loss):

    def __init__(self, n_simulation):
        super().__init__(n_simulation)

    #implemented
    def loss_function(self, prediction, observe):
        return np.sum(np.absolute(prediction - observe))

    #implemented
    def set_to_orginial_units(self, loss):
        return loss

    #implemented
    def get_short_name():
        return "mae"

    #implemented
    def get_short_bias_name():
        return "mab"

    #implemented
    def get_axis_label():
        return "mean absolute error (mm)"

    #implemented
    def get_axis_bias_label():
        return "mean absolute bias (mm)"

class MeanAbsolute10Error(MeanAbsoluteError):

    def __init__(self, n_simulation):
        super().__init__(n_simulation)

    #override
    def add_data(self, forecast, observed_data):
        #only consider observed data with precipitation greater than 10 mm
        is_above_10 = observed_data >= 10
        if np.any(is_above_10):
            super().add_data(forecast, observed_data, is_above_10)

    #override
    def get_short_name():
        return "mae10"

    #override
    def get_short_bias_name():
        return "mab10"

    #override
    def get_axis_label():
        return "mean absolute error 10 (mm)"

    #override
    def get_axis_bias_label():
        return "mean absolute bias 10 (mm)"
