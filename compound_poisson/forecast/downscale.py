import datetime
import math
import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import compound_poisson
from compound_poisson import roc
from compound_poisson.forecast import time_series
from compound_poisson.forecast import forecast_abstract

class Forecaster(forecast_abstract.Forecaster):
    """Contain Monte Carlo forecasts for Downscale

    Attributes:
        downscale: Downscale object, to forecast from
        data: the test set (dataset.Data object)
        forecast_array: memmap of forecasts
            dim 0: for each (unmasked) location
            dim 1: for each simulation
            dim 2: for each time point
    """

    def __init__(self, downscale, memmap_dir):
        self.downscale = downscale
        self.data = None
        super().__init__(memmap_dir)

    def make_memmap_path(self):
        super().make_memmap_path(type(self.downscale).__name__)

    def start_forecast(self, n_simulation, data):
        """Start forecast simulations, to be called initially

        Args:
            n_simulation: number of simulations
            data: the test set (dataset.Data object)
        """
        self.data = data
        self.n_time = len(data)
        self.time_array = data.time_array
        super().start_forecast(n_simulation)

    def copy_to_memmap(self, memmap_to_copy):
        for i in range(len(self.forecast_array)):
            memmap_to_copy_i = memmap_to_copy[i]
            self.forecast_array[i, 0:len(memmap_to_copy_i)] = memmap_to_copy_i

    def simulate_forecasts(self, index_range, is_print=True):
        area_unmask = self.downscale.area_unmask
        n_total_parameter = self.downscale.n_total_parameter

        forecast_message = []
        for i_space, time_series in enumerate(
            self.downscale.generate_unmask_time_series()):
            #extract model fields for each unmasked time_series
            lat_i = time_series.id[0]
            long_i = time_series.id[1]
            x_i = self.data.get_model_field(lat_i, long_i)
            #extract mcmc chain corresponding to this location
            parameter_mcmc = self.downscale.parameter_mcmc[:]
            parameter_mcmc = parameter_mcmc[
                :, range(i_space, n_total_parameter, area_unmask)]
            time_series.parameter_mcmc = parameter_mcmc

            message = ForecastMessage(time_series,
                                      x_i,
                                      self.n_simulation,
                                      self.memmap_path,
                                      self.forecast_array.shape,
                                      i_space,
                                      is_print)
            forecast_message.append(message)

        time_series_array = self.downscale.pool.map(
            ForecastMessage.forecast, forecast_message)
        self.downscale.replace_unmask_time_series(time_series_array)

    def get_prob_rain(self, rain, index=None):
        """Get the probability if it will rain at least of a certian amount

        Args:
            rain: scalar, amount of rain to evaluate the probability
            index: time index (optional), otherwise, take ROC from all time
                points

        Return:
            matrix, dim 0 for each location, dim 1 for each time step
        """
        if index is None:
            index = slice(self.n_time)
        #for forecast_array...
            #dim 0 is location
            #dim 1 is for each simulation
            #dim 2 is for each time point
        p_rain = np.mean(self.forecast_array[:, :, index] > rain, 1)
        return p_rain

    def load_memmap(self, mode):
        super().load_memmap(
            mode, (self.downscale.area_unmask, self.n_simulation, self.n_time))

    def generate_time_series_forecaster(self):
        """Generate the forecaster for every unmasked time series
        """
        for time_series in self.downscale.generate_unmask_time_series():
            forecaster = time_series.forecaster
            forecaster.load_memmap("r")
            yield time_series.forecaster

    def get_roc_curve_array(
        self, rain_warning_array, test_set, time_index=None):
        """Get ROC curves for range of rain warnings

        Args:
            rain_warning_array: array, containing amount of precipitation to
                detect
            test_set: dataset.Data object
            time_index: optional, a pointer (eg slice or array of indices) for
                time points to take ROC curve of

        Return:
            array of roc.Roc objects which can be None if a value of
                precipitation in rain_warning_array was never observed
        """
        if time_index is None:
            time_index = slice(len(test_set))
        mask = test_set.mask
        observed_rain = test_set.rain[time_index, np.logical_not(mask)]
        #swap axes so that...
            #dim 0: for each location
            #dim 1: for each time point
        observed_rain = np.swapaxes(observed_rain, 0, 1)
        #when flatten, this is comparable with the return value from
            #self.get_prob_rain()
        observed_rain = observed_rain.flatten()

        roc_array = []
        #get roc curve for every rain_warning, else None if that amount of rain
            #was never observed in the test set
        for rain_warning in rain_warning_array:
            if np.any(rain_warning < observed_rain):
                p_rain = self.get_prob_rain(rain_warning, time_index).flatten()
                roc_curve = roc.Roc(rain_warning, p_rain, observed_rain)
                roc_array.append(roc_curve)
            else:
                roc_array.append(None)
        return roc_array

class ForecasterDual(Forecaster):
    """Forecaster used by DownscaleDual

    Extended to be able to use samples from the MCMC. Forecasting for
        DownscaleDual is different compared to Downscale, this is because model
        fields of the test set needs to be sampled. All model fields for all
        time points are sampled altogether because they have a spatial
        correlation.
    For a forecast sample, all spatial parameters and training set model fields
        are set from the same MCMC sample. All test set model fields are
        sampled, parallel in time. Then for each location, in parallel, produce
        one forecast. This is repeated until a requested number of forecasts is
        achieved.

    Attributes:
        rng: spawned rng to sample from the MCMC samples
    """

    def __init__(self, downscale, memmap_dir):
        super().__init__(downscale, memmap_dir)
        #rng used so that all time series in the forecast use the same mcmc
            #sample
        self.rng = downscale.spawn_rng()

    def simulate_forecasts(self, index_range):
        """Simulate forecasts

        Override. Set all parameters and model fields from the mcmc sample, then
            do one forecast. This is repeated for all pointers in index_range.

        Args:
            index_range: array of pointers to save results onto forecast_array
        """
        area_unmask = self.downscale.area_unmask
        n_model_field = self.downscale.n_model_field
        model_field_mcmc = self.downscale.model_field_mcmc
        gp_mcmc = self.downscale.model_field_gp_mcmc
        n_mcmc_sample = len(model_field_mcmc)

        #instantiate a DownscaleDual to do gaussian process regression and
            #sampling
        future_downscale = compound_poisson.DownscaleDual(
            self.data, self.downscale.n_arma, True) #true to show test data
        future_downscale.pool = self.downscale.pool

        for i_forecast in index_range:
            #force to do one forecast for every mcmc sample
            self.n_simulation = i_forecast+1

            #set parameters and model fields from a mcmc sample
                #the setting of the parameters is done in the super class method
            mcmc_index = self.rng.randint(self.downscale.burn_in, n_mcmc_sample)
            self.downscale.set_mcmc_index(mcmc_index)
            model_field_sample = model_field_mcmc[mcmc_index]
            gp_sample = gp_mcmc[mcmc_index]
            self.downscale.model_field_target.update_state(model_field_sample)

            #sample test set model field GP
            future_downscale.model_field_gp_target.update_state(gp_sample)
            model_field_sample = (
                future_downscale.model_field_target.simulate_from_prior(
                    self.rng))

            #see mcmc/target_model_field.py for the layout of the vector
                #model_field_sample
            #change the test data to have the model field GP sample
            for i_time in range(len(future_downscale)):
                for i_model_field, model_field_name in enumerate(
                    self.data.model_field):
                    for i_space, (lat_i, long_i) in enumerate(
                        self.data.generate_unmask_coordinates()):
                        x_i = model_field_sample[
                            i_time*n_model_field*area_unmask
                            + i_model_field*area_unmask
                            + i_space]
                        model_field_i = self.data.model_field[model_field_name]
                        model_field_i[i_time, lat_i, long_i] = x_i

            super().simulate_forecasts([i_forecast], False)
            print("Predictive sample", i_forecast)

class TimeSeriesForecaster(time_series.Forecaster):
    """Used by TimeSeriesDownscale class

    Extended to handle MCMC samples in memmaps. The member variable
        forecast_array is shared with all spatial points and is handled by the
        corresponding Downscale object.

    Attributes:
        i_space: pointer for space, or the 0th dimension for the forecast_array
        mememap_path: location of the forecast_array
        memmap_shape: shape of the forecast_array
    """

    def __init__(self, time_series, memmap_path, i_space):
        super().__init__(time_series, path.dirname(memmap_path))
        self.i_space = i_space
        self.memmap_path = memmap_path
        self.memmap_shape = None

    def start_forecast(self, n_simulation, model_field, memmap_shape):
        """Start forecast simulations, to be called initially

        Override as a memmap does not need to be created to store the forecasts,
            this has already been done by the corresponding Downscale object.

        Args:
            n_simulation: number of simulations
            model_field: model fields for test set
            memmap_shape: shape of the forecast_array
        """
        self.memmap_shape = memmap_shape
        self.model_field = model_field
        super().start_forecast(n_simulation, model_field)

    def make_memmap_path(self):
        """Do nothing, memmap_path has already been provided
        """
        pass

    def simulate_forecasts(self, index_range):
        #do not print progress
        super().simulate_forecasts(index_range, False)

    def resume_forecast(self, n_simulation, memmap_shape):
        """Simulate more forecasts

        Override as the handling of forecast_array is done already by the
            corresponding Downscale object.

        Args:
            n_simulation: total amount of simulations, ie should be higher than
                previous
            memmap_shape: shape of the forecast_array
        """
        if n_simulation > self.n_simulation:
            self.memmap_shape = memmap_shape
            n_simulation_old = self.n_simulation
            self.n_simulation = n_simulation
            self.load_memmap("r+")
            #False in argument to not print progress
            self.simulate_forecasts(range(n_simulation_old, self.n_simulation))
            self.del_memmap()

    def load_memmap(self, mode):
        """Load the memmap file for forecast_array

        Override to use the forcast_array provided by Downscale. This is shared
            with all spatial points so extract the corresponding slice.

        Args:
            mode: how to read the memmap file, eg "w+", "r+", "r"
        """
        super().load_memmap(mode, self.memmap_shape)
        self.forecast_array = self.forecast_array[self.i_space]

class ForecastMessage(object):
    """Message to forecast all spatial points in parallel
    """
    def __init__(self,
                 time_series,
                 model_field,
                 n_simulation,
                 memmap_path,
                 memmap_shape,
                 i_space,
                 is_print):
        self.time_series = time_series
        self.model_field = model_field
        self.n_simulation = n_simulation
        self.memmap_path = memmap_path
        self.memmap_shape = memmap_shape
        self.i_space = i_space
        self.is_print = is_print

    def forecast(self):
        self.time_series.forecast(self.model_field, self.n_simulation,
            self.memmap_path, self.memmap_shape, self.i_space)
        if self.is_print:
            print("Predictive location", self.i_space)
        return self.time_series
