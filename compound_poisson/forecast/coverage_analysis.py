"""For the analysis of credible interval widths (aka HDI) and the coverage, the
    proportion of observed data points in the credible intervals.

How to use:
    Instantiate a compound_poisson.forecast.time_segmentation.TimeSegmentator
        object. This shall be pass in the contructors.
    Instantiate TimeSeries or Downscale, call the method add_data() and extract
        results from the member variables.
    The member variable credible_level_array may be modified for evaluation of
        different credible levels.

TimeSeries <- Downscale
"""

import numpy as np

from compound_poisson.forecast import time_segmentation

class TimeSeries(object):
    """For analysising credible interval widths and coverage for TimeSeries

    Attributes:
        credible_level_array: array of credible intervals to evaluate
        coverage_array: coverage for each credible level and time point in
            time_segmentator
            dim 0: for each credible level
            dim 1: for each time point
        spread_array: (average) widths of the credible intervals
            dim 0: for each credible level
            dim 1: for each time point
        time_array: every time point in time_segmentator
        time_segmentator: TimeSegmentator object
    """

    def __init__(self, time_segmentator):
        """
        Args:
            time_segmentator: TimeSegmentator object
        """
        self.credible_level_array = np.array([0.5, 0.68, 0.95, 0.99])
        self.coverage_array = []
        self.spread_array = []
        self.time_array = time_segmentator.get_time_array()
        self.time_segmentator = time_segmentator

    def add_data(self, forecaster, observed_rain):
        """Add a TimeSeries data, update the member variables coverage_array
            and spread_array

        Args:
            forecaster: compound_poisson.forecast.time_series.Forecaster object
            observed_rain: numpy array of observed precipitation
        """
        self.coverage_array, self.spread_array = self.get_coverage_time_series(
            forecaster, observed_rain)

    def get_coverage_time_series(self, forecaster, observed_rain):
        """Evaluates the average spread and coverage for a TimeSeries

        Args:
            forecaster: compound_poisson.forecast.time_series.Forecaster object
            observed_rain: numpy array of observed precipitation
        Returns:
            coverage_array: coverage for each credible level and time point in
                time_segmentator
                dim 0: for each credible level
                dim 1: for each time point
            spread_array: (average) widths of the credible intervals
                dim 0: for each credible level
                dim 1: for each time point
        """
        #in this method only, coverage_array and spread_array has dimensions:
            #dim 0: for each time
            #dim 1: for each credible interval
        #the returned coverage_array is transposed (dimensions swapped)
        #same with spread_array
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

            #array of coverage and spread for this time, each elemenet
                #correspond to different credible levels
            coverage_i = []
            spread_i = []
            #for each credible level, evaluate the mean spread and coverage
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
        #transpose which swap the dimensions
        coverage_array = np.asarray(coverage_array).T
        spread_array = np.asarray(spread_array).T
        return (coverage_array, spread_array)

class Downscale(TimeSeries):
    """For analysising credible interval widths and coverage for Downscale
    """

    def __init__(self, time_segmentator):
        super().__init__(time_segmentator)

    #override
    def add_data(self, forecaster):
        """Add a TimeSeries data, update the member variables coverage_array
            and spread_array

        Args:
            forecaster: compound_poisson.forecast.downscale.Forecaster object
        """
        #coverage_array and spread_array at the start of this method:
            #dim 0: for each location
            #dim 1: for each credible interval
            #dim 2: for each time
        self.coverage_array = []
        self.spread_array = []
        #for each location, work out coverage and mean spread
        iter_rain = zip(forecaster.generate_time_series_forecaster(),
                        forecaster.data.generate_unmask_rain())
        for forecaster_i, observed_rain_i in iter_rain:
            coverage_i, spread_i = self.get_coverage_time_series(
                forecaster_i, observed_rain_i)
            self.coverage_array.append(coverage_i)
            self.spread_array.append(spread_i)
            forecaster_i.del_memmap()
        #average over locations
        self.coverage_array = np.asarray(self.coverage_array)
        self.spread_array = np.asarray(self.spread_array)
        self.coverage_array = np.mean(self.coverage_array, 0)
        self.spread_array = np.mean(self.spread_array, 0)
