import numpy as np

import compound_poisson

class TimeSeries(compound_poisson.time_series.TimeSeries):

    def __init__(self):
        #pass a dummy x to instantiate a blank TimeSeries
        super().__init__(np.ones((2,1)))

    def fit(self, forecast_future, time_array=None):
        """Wrap around the ERA5 data
        """
        self.forecaster = ForecasterTimeSeries(
            self, forecast_future, time_array)

class ForecasterTimeSeries(compound_poisson.forecast.time_series.Forecaster):

    def __init__(self, time_series, forecast_future, time_array=None):
        """emulates the following overloads:
        Args:
            forecast_future: a Era5 object
            time_array: not used
        Args:
            forecast_future: numpy array of forecasted precipitation
            time_array: array of dates
        """
        super().__init__(time_series, None)
        if time_array is None:
            self.time_array = forecast_future.time_array
            self.forecast = forecast_future.rain
        else:
            self.time_array = time_array
            self.forecast = forecast_future

        self.forecast_median = self.forecast
        self.forecast_array = np.asarray([self.forecast])
        self.n_simulation = 1
        self.n_time = len(self.time_array)

    def load_memmap(self, mode, memmap_shape=None):
        #override to do nothing
        pass

    def del_memmap(self):
        #override to do nothing
        pass

class Downscale(compound_poisson.downscale.Downscale):

    def __init__(self, forecast_future):
        super().__init__(forecast_future)

    def fit(self, forecast_future, test_set):
        """Wrap around the ERA5 data (and test set data, required for
            comparison)
        """
        self.forecaster = ForecasterDownscale(self, forecast_future, test_set)

    def get_time_series_class(self):
        #override so that each location is represented using a
            #compound_poisson.era5.TimeSeries object
        return TimeSeries

class ForecasterDownscale(compound_poisson.forecast.downscale.Forecaster):

    def __init__(self, downscale, forecast_future, test_set):
        #reminder about forecast_array:
            #dim 0: for each (unmasked) location
            #dim 1: for each simulation
            #dim 2: for each time point
        super().__init__(downscale, None)
        self.data = test_set
        self.time_array = forecast_future.time_array
        self.forecast_array = []
        self.n_simulation = 1
        self.n_time = len(self.time_array)
        #scatter the forecast to each location
        for time_series_i in self.downscale.generate_unmask_time_series():
            lat_i = time_series_i.id[0]
            long_i = time_series_i.id[1]
            forecast_i = forecast_future.rain[:, lat_i, long_i]
            self.forecast_array.append([forecast_i])
            time_series_i.fit(forecast_i, self.time_array)
        self.forecast_array = np.asarray(self.forecast_array)

    def load_memmap(self, mode, memmap_shape=None):
        #override to do nothing
        pass

    def del_memmap(self):
        #override to do nothing
        pass
