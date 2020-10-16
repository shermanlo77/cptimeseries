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
        self.bias_median_loss_sum = 0
        self.n_time_points = 0

    def add_data(self, forecast, observed_data, index=None):
        """Update member variables (eg losses) with new data (for a single
            location). For multiple locations, see add_downscale_forecaster()

        Args:
            forecast: forecaster.time_series.TimeSeries object
            observed_data: numpy array of observed precipitation, same time
                length as forecast
            index: array of booleans, only consider data with True
        """

        forecast_expectation = None
        forecast_median = None
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
            forecaster_i_og = forecaster_i
            if not index is None:
                forecaster_i = forecaster_i[index]
                observed_rain_i = observed_rain_i[index]
            self.add_data(forecaster_i, observed_rain_i)

    def get_bias_loss(self):
        """Return a SINGLE forecast (eg mean over samples) with the obserbed
            metric, eg root mean bias squared
        """
        if self.n_time_points == 0:
            return math.nan
        else:
            return self.set_to_orginial_units(
                self.bias_loss_sum / self.n_time_points)

    def get_bias_median_loss(self):
        """Return a SINGLE forecast (eg median over samples) with the obserbed
            metric, eg root mean bias squared
        """
        if self.n_time_points == 0:
            return math.nan
        else:
            return self.set_to_orginial_units(
                self.bias_median_loss_sum / self.n_time_points)

    def get_risk(self):
        """Return the posterior expectation of the loss, aka the risk
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
        """Return the sum of losses
        """
        raise NotImplementedError

    def set_to_orginial_units(self, loss):
        """Normalise the loss to the units of the orginial data
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

    def set_to_orginial_units(self, loss):
        return np.sqrt(loss)

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
        is_above_10 = observed_data >= 10
        if np.any(is_above_10):
            super().add_data(forecast, observed_data, is_above_10)

    def get_short_name():
        return "rmse10"

    def get_short_bias_name():
        return "rmsb10"

    def get_axis_label():
        return "root mean square error 10 (mm)"

    def get_axis_bias_label():
        return "root mean square bias 10 (mm)"

class MeanAbsoluteError(Loss):

    def __init__(self, n_simulation):
        super().__init__(n_simulation)

    def loss_function(self, prediction, observe):
        return np.sum(np.absolute(prediction - observe))

    def set_to_orginial_units(self, loss):
        return loss

    def get_short_name():
        return "mae"

    def get_short_bias_name():
        return "mab"

    def get_axis_label():
        return "mean absolute error (mm)"

    def get_axis_bias_label():
        return "mean absolute bias (mm)"

class MeanAbsolute10Error(MeanAbsoluteError):

    def __init__(self, n_simulation):
        super().__init__(n_simulation)

    def add_data(self, forecast, observed_data):
        is_above_10 = observed_data >= 10
        if np.any(is_above_10):
            super().add_data(forecast, observed_data, is_above_10)

    def get_short_name():
        return "mae10"

    def get_short_bias_name():
        return "mab10"

    def get_axis_label():
        return "mean absolute error 10 (mm)"

    def get_axis_bias_label():
        return "mean absolute bias 10 (mm)"
