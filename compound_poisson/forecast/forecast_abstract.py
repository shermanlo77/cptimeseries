import datetime
import os
from os import path

import numpy as np

class Forecaster(object):
    """Contain Monte Carlo forecasts

    Abstract class
    Contain Monte Carlo forecasts in forecast_array as a numpy.memmap object.
        Add forecasts using the method start_forecast(). To add more forecasts
        after that, call the method resume_forecast(). Summary statistics are
        stored in member variables.
    Used by classes TimeSeries and Downscale when calling forecast() and
        forecast_self().

    Methods to implement: copy_to_memmap(), simulate_forecasts(),
        get_prob_rain()
    Methods to override: load_memmap() by providing memmap_shape,
        start_forecast() by adding an extra argument to accept data,
        make_memmap_path() by providing a

    Attributes:
        time_array: array, containing time stamps for each point in the forecast
        forecast_array: memmap of forecasts, shape depends on subclass
            implementation
        n_time: length of forecast or number of time points
        n_simulation: number of simulations to be done
        memmap_dir: directory to forecast_array memmap file
        memmap_path: file path to forecast_array memmap file
    """

    def __init__(self, memmap_dir):
        self.time_array = None
        self.forecast_array = None
        self.n_time = None
        self.n_simulation = 0
        self.memmap_dir = memmap_dir
        self.memmap_path = None

    def make_memmap_path(self, description):
        """Make a new memmap file name

        Uses datetime to make it unique and identifiable. Should be overriden
            by providing the description for the memmap file name.

        Args:
            description: used to name the file
        """
        datetime_id = str(datetime.datetime.now())
        datetime_id = datetime_id.replace("-", "")
        datetime_id = datetime_id.replace(":", "")
        datetime_id = datetime_id.replace(" ", "")
        datetime_id = datetime_id[0:14]
        file_name = ("_" + type(self).__name__ + description
            + datetime_id + ".dat")
        self.memmap_path = path.join(self.memmap_dir, file_name)

    def start_forecast(self, n_simulation):
        """Start forecast simulations, to be called initially

        Should be overriden by adding an extra argument to accept data, handles
            it, then call this class's method to do forecasting.

        Args:
            n_simulation: number of simulations
            data: test set (or training set) data (providing None indicates to
                use training set)
        """
        self.n_simulation = n_simulation
        self.make_memmap_path()
        self.load_memmap("w+")
        self.simulate_forecasts(range(n_simulation))
        self.del_memmap()

    def resume_forecast(self, n_simulation):
        """Simulate more forecasts

        Args:
            n_simulation: total amount of simulations, ie should be higher than
                previous
        """
        if n_simulation > self.n_simulation:
            n_simulation_old = self.n_simulation
            #transfer forecasts from old file to new file
            memmap_path_old = self.memmap_path
            self.load_memmap("r")
            forecast_array_old = self.forecast_array
            self.n_simulation = n_simulation
            self.make_memmap_path()
            self.load_memmap("w+")
            self.copy_to_memmap(forecast_array_old)
            del forecast_array_old
            #simulate more forecasts
            self.simulate_forecasts(range(n_simulation_old, self.n_simulation))
            if path.exists(memmap_path_old):
                os.remove(memmap_path_old)
            self.del_memmap()

    def copy_to_memmap(self, memmap_to_copy):
        """Copy a given array to the forecast_array memmap

        Copy a given array to the forecast_array memmap. The given array may
            contain less samples than forecast_array.
        """
        raise NotImplementedError

    def simulate_forecasts(self, index_range, is_print=True):
        """Simulate forecasts

        Simulate the forecasts and save results to self.forecast_array

        Args:
            index_range: array of pointers to save results onto forecast_array
                eg [0,1,2,3] for the first 4 inital forecasts, then [4,5,6] for
                3 more
            is_print: optional, True if to print progress of forecasting
        """
        raise NotImplementedError

    def get_prob_rain(self, rain, index=None):
        """Get the probability if it will rain at least of a certian amount

        Args:
            rain: scalar, amount of rain to evaluate the probability
            index: time index (optional), otherwise, take ROC from all time
                points

        Return:
            vector or matrix of probabilities, see subclasses
        """
        raise NotImplementedError

    def load_memmap(self, mode, memmap_shape):
        """Load the memmap file for forecast_array

        To be overriden by providing memmap_shape

        Args:
            mode: how to read the memmap file, eg "w+", "r+", "r"
            memmap_shape: the shape of forecast_array
        """
        self.forecast_array = np.memmap(self.memmap_path,
                                        np.float64,
                                        mode,
                                        shape=memmap_shape)

    def del_memmap(self):
        """Delete the file handling
        """
        del self.forecast_array
        self.forecast_array = None
