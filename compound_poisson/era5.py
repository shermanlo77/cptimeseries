import numpy as np

import compound_poisson

class TimeSeries(compound_poisson.time_series.TimeSeries):

    def __init__(self):
        #pass a dummy x to instantiate a blank TimeSeries
        super().__init__(np.ones((2,1)))

    def fit(self, forecast_future):
        """Wrap around the ERA5 data
        """
        self.forecaster = ForecasterTimeSeries(self, forecast_future)

class ForecasterTimeSeries(compound_poisson.forecast.time_series.Forecaster):

    def __init__(self, time_series, forecast_future):
        super().__init__(time_series, None)
        self.time_array = forecast_future.time_array
        self.forecast = forecast_future.rain
        self.forecast_median = self.forecast
        self.forecast_array = np.asarray([self.forecast])
        self.n_simulation = 1

    def load_memmap(self, mode, memmap_shape=None):
        #override to do nothing
        pass

    def del_memmap(self):
        #override to do nothing
        pass
