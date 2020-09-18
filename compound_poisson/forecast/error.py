import math

from matplotlib import pyplot as plt
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
            self.sum_square += np.sum(np.square(forecast_10 - observed_10))

    def get_error(self):
        #handle sistuations where it never rained more than 10 mm
        if self.n == 0:
            return math.nan
        else:
            return super().get_error()

    def get_short_name():
        return "r10"

    def get_axis_label():
        return "rmse 10 (mm)"

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
        return "mean absolute error (mm)"

class ResidualHist(Error):

    def __init__(self):
        super().__init__()
        self.residual = np.array([])
        self.observed_data = np.array([])

    def add_data(self, forecast, observed_data):
        self.residual = np.concatenate(
            (self.residual, forecast.forecast_median - observed_data))
        self.observed_data = np.concatenate(
            (self.observed_data, observed_data))

    def get_error(self):
        [hist, x_edges, y_edges] = np.histogram2d(
            self.observed_data, self.residual, 30, density=True)
        hist[hist==0] = hist[hist>0].min()
        hist *= len(self.residual)
        plt.figure()
        plt.pcolormesh(x_edges, y_edges, np.log10(hist).T)
        plt.xlabel("observed precipitation (mm)")
        plt.ylabel("residual (mm)")
        plt.colorbar()
