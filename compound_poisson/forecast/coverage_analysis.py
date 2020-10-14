import numpy as np

from compound_poisson.forecast import time_segmentation

class TimeSeries(object):

    def __init__(self, time_segmentator):
        #coverage_array:
            #dim 0: for each quantile
            #dim 1: for each time point
        self.credible_level_array = np.array([0.5, 0.68, 0.95, 0.99])
        self.coverage_array = []
        self.spread_array = []
        self.time_array = []
        self.time_segmentator = time_segmentator
        for date, index in self.time_segmentator:
            self.time_array.append(date)

    def add_data(self, forecaster, observed_rain):
        self.coverage_array, self.spread_array = self.get_coverage_time_series(
            forecaster, observed_rain)

    def get_coverage_time_series(self, forecaster, observed_rain):
        #in this method only, coverage_array and spread_array has dimensions:
            #dim 0: for each time
            #dim 1: for each credible interval
        #returns coverage_array but transposed (dimensions swapped), same with
            #spread_array
        coverage_array = []
        spread_array = []
        for date, index in self.time_segmentator:
            forecast_slice = forecaster[index]
            observed_slice = observed_rain[index]
            lower_p = (1 - self.credible_level_array) / 2
            upper_p = (1 + self.credible_level_array) / 2
            lower_error_array = np.quantile(
                forecast_slice.forecast_array, lower_p, 0)
            upper_error_array = np.quantile(
                forecast_slice.forecast_array, upper_p, 0)

            coverage_i = []
            spread_i = []
            for lower_error_j, upper_error_j in zip(
                lower_error_array, upper_error_array):
                #equality required to compare with 0 mm
                coverage_ij = np.mean(
                    np.logical_and(observed_slice >= lower_error_j,
                                   observed_slice <= upper_error_j))
                spread_ij = np.mean(upper_error_j - lower_error_j)
                coverage_i.append(coverage_ij)
                spread_i.append(spread_ij)
            coverage_array.append(coverage_i)
            spread_array.append(spread_i)
        coverage_array = np.asarray(coverage_array).T
        spread_array = np.asarray(spread_array).T
        return (coverage_array, spread_array)

class Downscale(TimeSeries):

    def __init__(self, time_segmentator):
        super().__init__(time_segmentator)

    def add_data(self, forecaster, observed_data):
        #coverage array:
            #dim 0: for each location
            #dim 1: for each credible interval
            #dim 2: for each time
        self.coverage_array = []
        self.spread_array = []
        iter_rain = zip(forecaster.downscale.generate_unmask_time_series(),
                        observed_data.generate_unmask_rain())
        for time_series_i, observed_rain_i in iter_rain:
            coverage_i, spread_i = self.get_coverage_time_series(
                time_series_i.forecaster, observed_rain_i)
            self.coverage_array.append(coverage_i)
            self.spread_array.append(spread_i)
        #average over locations
        self.coverage_array = np.asarray(self.coverage_array)
        self.spread_array = np.asarray(self.spread_array)
        self.coverage_array = np.mean(self.coverage_array, 0)
        self.spread_array = np.mean(self.spread_array, 0)
