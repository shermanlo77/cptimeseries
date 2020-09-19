"""Classes for plotting observed precipitation with the forecast (or a function
    of them)
"""

from matplotlib import pyplot as plt
import numpy as np

HIST_BINS = 30 #number of bins for 2d histogram

class ResidualPlotter(object):
    """Abstract class for plotting residuals

    How to use or implement:
        Call add_data() to input data. Can be called multiple times to add data
            from different locations or times
        Call plot_scatter() or plot_heatmap() to plot figures. The x and y axis
            are to be implemented in get_x_data() and get_y_data()

    Attributes:
        forecast: array of forecasts
        observed_data: array of observed precipitation, same length as forecast
    """

    def __init__(self, residual_plotter=None):
        """Constructor, pass residual_plotter for a (shallow) copy or a
            (shallow) cast
        """
        self.forecast = None
        self.observed_data = None
        if residual_plotter is None:
            self.forecast = np.array([])
            self.observed_data = np.array([])
        else:
            self.forecast = residual_plotter.forecast
            self.observed_data = residual_plotter.observed_data

    def add_data(self, forecast, observed_data):
        self.forecast = np.concatenate(
            (self.forecast, forecast.forecast_median))
        self.observed_data = np.concatenate(
            (self.observed_data, observed_data))

    def get_x_data(self):
        raise NotImplementedError

    def get_y_data(self):
        raise NotImplementedError

    def get_x_label(self):
        return ""

    def get_y_label(self):
        return ""

    def plot_scatter(self):
        """Scatter plot
        """
        x_data = self.get_x_data()
        y_data = self.get_y_data()
        plt.figure()
        plt.scatter(x_data, y_data, 2)
        plt.xlabel(self.get_x_label())
        plt.ylabel(self.get_y_label())

    def plot_heatmap(self):
        """Log frequency density plot
        """
        #area x 10^colour = count (aka frequency)
        x_data = self.get_x_data()
        y_data = self.get_y_data()
        plt.figure()
        [hist, x_edges, y_edges] = np.histogram2d(
            x_data, y_data, HIST_BINS, density=True)
        hist[hist==0] = hist[hist>0].min()
        hist *= len(x_data)
        plt.pcolormesh(x_edges, y_edges, np.log10(hist).T)
        plt.xlabel(self.get_x_label())
        plt.ylabel(self.get_y_label())
        plt.colorbar()

class ResidualBaPlotter(ResidualPlotter):
    """Plots residual vs observed data
    Ba stands for Bland–Altman (it isn't exactly Bland-Altman but it is like)
    """

    def __init__(self, residual_plotter=None):
        super().__init__(residual_plotter)

    def get_x_data(self):
        return self.observed_data

    def get_y_data(self):
        return self.forecast - self.observed_data

    def get_x_label(self):
        return "observed precipitation (mm)"

    def get_y_label(self):
        return "residual (mm)"

class ResidualLnqqPlotter(ResidualPlotter):
    """Plots ln(forecast+1) vs ln(observed+1)
    Lnqq stands for log natural quantile-quantile
    """

    def __init__(self, residual_plotter=None):
        super().__init__(residual_plotter)

    def get_x_data(self):
        return np.log(self.observed_data + 1)

    def get_y_data(self):
        return np.log(self.forecast + 1)

    def get_x_label(self):
        return "observed precipitation (ln mm)"

    def get_y_label(self):
        return "residual (ln mm)"
