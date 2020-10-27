"""Classes for printing out figures for the resulting forecast

How to use:
    Instantiate TimeSeries or Downscale then call the method print()

Base class is Printer
    Implemented by TimeSeries and Downscale

Downscale:
    Uses multiple instantiations of PrintForecastMapMessage and
        PrintForecastSeriesMessage
"""

import datetime
import math
import os
from os import path

from cartopy import crs
import cycler
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import pandas.plotting
from scipy import stats

import compound_poisson
from compound_poisson.forecast import coverage_analysis
from compound_poisson.forecast import loss
from compound_poisson.forecast import loss_segmentation
from compound_poisson.forecast import residual_analysis
from compound_poisson.forecast import roc
from compound_poisson.forecast import time_segmentation
import dataset

RAIN_THRESHOLD_ARRAY = [0, 5, 10, 15] #rain to consider for yearly ROC
RAIN_THRESHOLD_EXTREME_ARRAY = [5, 15, 25] #rain to consider for total ROC
LINESTYLE = ['-', '--', '-.', ':']

def get_monochrome_cycler():
    #return cycler for monochrome plotting
    return cycler.cycler('color', ['k']) * cycler.cycler('linestyle', LINESTYLE)

class Printer(object):
    """Abstract class for printing out figures for TimeSeries and Downscale

    How to use: instantiate and call the method print()

    Attributes:
        forecaster: compound_poisson.forecast.forecast_abstract.Forecaster
            object
        directory: where to save the figures
        prefix: what to put as a prefix for the file names for the figures
    """

    def __init__(self, forecaster, directory, prefix=""):
        """
        Args:
            forecaster: compound_poisson.forecast.forecast_abstract.Forecaster
                object
            directory: where to save the figures
            prefix: what to put as a prefix for the file names for the figures
        """
        self.forecaster = forecaster
        self.directory = directory
        self.prefix = prefix

    def print(self):
        """Print out all of the figures, these include:
            -Forecasts, see print_forecast()
            -AUC for each year
            -ROC curves for each year
            -ROC curves for the entire test set
            -Survival, pp and qq plot comparing the distributions of the
                predictive with the observed
            -Bias loss for each time segment (eg every winter or every year)
            -Residual plots:
                -residual vs observed
                -ln(forecast+1) vs ln(observed+1)
            -Coverage of each year for different credible levels (aka HDI)
            -Interval width (aka spread) vs coverage

        Shall be responsible for load_memmap() and del_memmap() calls
        """

        self.forecaster.load_memmap("r")
        plt.rcParams.update({'font.size': 14})
        pandas.plotting.register_matplotlib_converters()
        monochrome = get_monochrome_cycler()

        time_array = self.forecaster.time_array
        year_segmentator = time_segmentation.YearSegmentator(time_array)
        spring_segmentator = time_segmentation.SpringSegmentator(time_array)
        summer_segmentator = time_segmentation.SummerSegmentator(time_array)
        autumn_segmentator = time_segmentation.AutumnSegmentator(time_array)
        winter_segmentator = time_segmentation.WinterSegmentator(time_array)

        #main forecasts plot
        self.print_forecast()

        #####-----ROC ANALYSIS-----######

        date_array = year_segmentator.get_time_array()
        #dictionary of AUC for different thresholds:
            #key: amount of rain,
            #value: array containing AUC for each year
        auc_array = {}
        for rain in RAIN_THRESHOLD_ARRAY:
            auc_array[rain] = []
        #plot roc curves for each year
        for date, index in year_segmentator:
            year = date.year
            #plot ROC curve, save AUC in auc_array
            plt.figure()
            ax = plt.gca()
            ax.set_prop_cycle(monochrome)
            roc_curve_array = self.get_roc_curve_array(
                RAIN_THRESHOLD_ARRAY, index)
            for i_rain, roc_curve in enumerate(roc_curve_array):
                if not roc_curve is None:
                    roc_curve.plot()
                    auc = roc_curve.area_under_curve
                else:
                    auc = np.nan
                auc_array[RAIN_THRESHOLD_ARRAY[i_rain]].append(auc)
            plt.legend(loc="lower right")
            plt.savefig(
                path.join(self.directory,
                          self.prefix + "roc_" + str(year) + ".pdf"),
                bbox_inches="tight")
            plt.close()

        #plot auc for each year
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(monochrome)
        for rain in RAIN_THRESHOLD_ARRAY:
            label = str(rain) + " mm"
            auc = np.asarray(auc_array[rain])
            is_number = np.logical_not(np.isnan(auc))
            date_array_plot = np.array(date_array)[is_number]
            plt.plot(date_array_plot, auc[is_number], label=label)
        plt.legend()
        plt.ylim([0.5, 1])
        plt.xlabel("year")
        plt.xticks(rotation=45)
        plt.ylabel("area under curve")
        plt.savefig(path.join(self.directory, self.prefix + "auc.pdf"),
                    bbox_inches="tight")
        plt.close()

        #plot roc curve for the entire test set
        roc_curve_array = self.get_roc_curve_array(RAIN_THRESHOLD_EXTREME_ARRAY)
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(monochrome)
        for roc_curve in roc_curve_array:
            if not roc_curve is None:
                roc_curve.plot()
        plt.legend(loc="lower right")
        plt.savefig(path.join(self.directory, self.prefix + "roc_all.pdf"),
                    bbox_inches="tight")
        plt.close()

        #####-----DISTIBUTION ANALYSIS-----######

        comparer = self.get_distribution_comparer()

        #survival plot
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(monochrome)
        comparer.plot_survival()
        comparer.adjust_survival_plot()
        plt.legend()
        plt.savefig(
            path.join(self.directory, self.prefix + "distribution.pdf"),
            bbox_inches="tight")
        plt.close()

        #pp plot
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(monochrome)
        comparer.plot_pp()
        comparer.adjust_pp_plot()
        plt.savefig(
            path.join(self.directory, self.prefix + "distribution_pp.pdf"),
            bbox_inches="tight")
        plt.close()

        #qq plot
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(monochrome)
        comparer.plot_qq()
        comparer.adjust_qq_plot()
        plt.savefig(
            path.join(self.directory, self.prefix + "distribution_qq.pdf"),
            bbox_inches="tight")
        plt.close()

        #####-----BIAS LOSS-----######
        #plot the loss for each time segment
        loss_segmentator = self.get_loss_segmentator()
        loss_segmentator.evaluate_loss(year_segmentator)
        loss_segmentator.plot_loss(
            self.directory, self.prefix+"yearly_", monochrome)
        loss_segmentator.evaluate_loss(spring_segmentator)
        loss_segmentator.plot_loss(
            self.directory, self.prefix+"spring_", monochrome)
        loss_segmentator.evaluate_loss(summer_segmentator)
        loss_segmentator.plot_loss(
            self.directory, self.prefix+"summer_", monochrome)
        loss_segmentator.evaluate_loss(autumn_segmentator)
        loss_segmentator.plot_loss(
            self.directory, self.prefix+"autumn_", monochrome)
        loss_segmentator.evaluate_loss(winter_segmentator)
        loss_segmentator.plot_loss(
            self.directory, self.prefix+"winter_", monochrome)

        #####-----RESIDUAL ANLYSIS-----######
        residual_plot = self.get_residual_analyser()

        #residual vs observed
        residual_plot.plot_heatmap(cmap="Greys")
        plt.savefig(path.join(self.directory,
                              self.prefix + "residual_hist.pdf"),
                    bbox_inches="tight")
        plt.close()

        residual_plot.plot_scatter()
        plt.savefig(path.join(self.directory,
                              self.prefix + "residual_scatter.png"),
                    bbox_inches="tight")
        plt.close()

        #forecast vs observed
        residual_plot = residual_analysis.ResidualLnqqPlotter(residual_plot)
        residual_plot.plot_heatmap(cmap="Greys")
        plt.savefig(path.join(self.directory,
                              self.prefix + "residual_qq_hist.pdf"),
                    bbox_inches="tight")
        plt.close()

        residual_plot.plot_scatter()
        plt.savefig(path.join(self.directory,
                              self.prefix + "residual_qq_scatter.png"),
                    bbox_inches="tight")
        plt.close()

        #####-----COVERAGE AND SPREAD ANLYSIS-----######

        #coverage vs year
        coverage = self.get_coverage_analyser()
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(monochrome)
        for i, credible_level in enumerate(coverage.credible_level_array):
            plt.plot(coverage.time_array,
                     coverage.coverage_array[i],
                     label=r"$\alpha = $" + str(credible_level))
        plt.xlabel("year")
        plt.xticks(rotation=45)
        plt.ylabel("coverage of HDI")
        plt.legend()
        plt.plot()
        plt.savefig(path.join(self.directory, self.prefix + "coverage.pdf"),
                    bbox_inches="tight")
        plt.close()

        #spread vs coverage
        coverage = self.get_spread_analyser()
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(monochrome)
        plt.plot(
            coverage.spread_array.flatten(), coverage.coverage_array.flatten())
        plt.xlabel("mean width of HDI (mm)")
        plt.ylabel("coverage of HDI")
        plt.plot()
        plt.savefig(path.join(self.directory, self.prefix + "spread.pdf"),
                    bbox_inches="tight")
        plt.close()

        #####-----CLEAN UP-----######
        self.forecaster.del_memmap()

    def print_forecast(self):
        """Print figures for the forecasts
        """
        raise NotImplementedError

    def get_roc_curve_array(self, rain_warning_array, index=None):
        """Return array of ROC curves for different precipitation

        Args:
            rain_warning_array: array of precipitation to evaluate the ROC curve
            index: optional, slice object to point which time points to use

        Return:
            array of compound_poisson.roc.Roc objects, one for each element in
                rain_warning_array
        """
        raise NotImplementedError

    def get_distribution_comparer(self):
        """Call and return the result from
            self.forecaster.compare_dist_with_observed()
        """
        raise NotImplementedError

    def get_loss_segmentator(self):
        """Returns an instantiated object from
            compound_poisson.forecast.loss_segmentation
        """
        raise NotImplementedError

    def get_residual_analyser(self):
        """Returns an instantiated object from
            compound_poisson.forecast.loss_segmentation
        """
        raise NotImplementedError

    def get_coverage_analyser(self):
        """Returns an instantiated object from
            compound_poisson.forecast.coverage_analysis. A YearSegmentator is
            used for yearly analysis.
        """
        raise NotImplementedError

    def get_spread_analyser(self):
        """Returns an instantiated object from
            compound_poisson.forecast.coverage_analysis. AllInclusive is
            used to investigate the entire test set. Lots of credible levels
            used so that spread vs coverage can be plotted
        """
        raise NotImplementedError

class TimeSeries(Printer):
    """For plotting forecast figures for TimeSeries

    Attributes:
        observed_rain: numpy array of observed daily precipitation
    """

    def __init__(self, forecaster, observed_rain, directory, prefix=""):
        """
        Args:
            forecaster: compound_poisson.forecast.time_series.Forecaster object
            observed_rain: numpy array of observed daily precipitation
            directory: where to save the figures
            prefix: what to put as a prefix for the file names for the figures
        """
        super().__init__(forecaster, directory, prefix)
        self.observed_rain = observed_rain

    #implemented
    def print_forecast(self):
        """Print figures for the forecasts

        The following figures are printed for each year:
            -mean forecast precipitation vs time + credible interval + observed
            -median forecast precipitation vs time + credible interval
                + observed
            -residual (using median) vs time
            -forecasted probability of precipitation great than x vs time for x
                in RAIN_THRESHOLD_ARRAY
        """

        monochrome = get_monochrome_cycler()
        colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        cycle_forecast = cycler.cycler(color=[colours[0], colours[1]],
                                       linewidth=[1, 1],
                                       alpha=[1, 0.5])

        year_segmentator = time_segmentation.YearSegmentator(
            self.forecaster.time_array)

        #for each year
        for date, index in year_segmentator:
            year = date.year

            #slice the current forecast and observation for this year
            forecast_sliced = self.forecaster[index]
            observed_rain_i = self.observed_rain[index]

            #get the forecasts
                #era5 won't have a sigma, handle accordingly
            forecast_i = forecast_sliced.forecast
            forecast_median_i = forecast_sliced.forecast_median
            if forecast_sliced.forecast_sigma:
                forecast_lower_error = forecast_sliced.forecast_sigma[-1]
                forecast_upper_error = forecast_sliced.forecast_sigma[1]
            else:
                forecast_lower_error = forecast_median_i
                forecast_upper_error = forecast_median_i
            time_array_i = forecast_sliced.time_array

            #plot forecast and observation time series
            plt.figure()
            ax = plt.gca()
            ax.set_prop_cycle(cycle_forecast)
            plt.fill_between(time_array_i,
                             forecast_lower_error,
                             forecast_upper_error,
                             alpha=0.25)
            plt.plot(time_array_i, forecast_i, label="forecast")
            plt.plot(time_array_i, observed_rain_i, label="observed")
            plt.xticks(rotation=45)
            plt.xlabel("time")
            plt.ylabel("precipitation (mm)")
            plt.legend()
            plt.savefig(
                path.join(self.directory,
                          self.prefix+"forecast_"+str(year)+".pdf"),
                bbox_inches="tight")
            plt.close()

            #plot forecast using median instead
            plt.figure()
            ax = plt.gca()
            ax.set_prop_cycle(cycle_forecast)
            plt.fill_between(time_array_i,
                             forecast_lower_error,
                             forecast_upper_error,
                             alpha=0.25)
            plt.plot(time_array_i, forecast_median_i, label="forecast")
            plt.plot(time_array_i, observed_rain_i, label="observed")
            plt.xticks(rotation=45)
            plt.xlabel("time")
            plt.ylabel("precipitation (mm)")
            plt.legend()
            plt.savefig(
                path.join(self.directory,
                          self.prefix+"forecast_median_"+str(year)+".pdf"),
                bbox_inches="tight")
            plt.close()

            #plot residual
            plt.figure()
            ax = plt.gca()
            ax.set_prop_cycle(monochrome)
            plt.plot(time_array_i, forecast_median_i - observed_rain_i)
            plt.xticks(rotation=45)
            plt.xlabel("time")
            plt.ylabel("residual (mm)")
            plt.savefig(
                path.join(self.directory,
                          self.prefix+"residual_"+str(year)+".pdf"),
                bbox_inches="tight")
            plt.close()

            #plot probability of more than rain precipitation
            for rain in RAIN_THRESHOLD_ARRAY:
                plt.figure()
                ax = plt.gca()
                ax.set_prop_cycle(monochrome)
                plt.plot(time_array_i, forecast_sliced.get_prob_rain(rain))
                plt.xticks(rotation=45)
                plt.xlabel("time")
                plt.ylabel("forecasted probability of > "+str(rain)+" mm")
                plt.savefig(
                    path.join(self.directory,
                              self.prefix+"prob_"+str(rain)+"_"+str(year)
                                +".pdf"),
                    bbox_inches="tight")
                plt.close()

    #implemented
    def get_roc_curve_array(self, rain_warning_array, index=None):
        return self.forecaster.get_roc_curve_array(
            rain_warning_array, self.observed_rain, index)

    #implemented
    def get_distribution_comparer(self):
        return self.forecaster.compare_dist_with_observed(self.observed_rain)

    #implemented
    def get_loss_segmentator(self):
        return loss_segmentation.TimeSeries(self.forecaster, self.observed_rain)

    #implemented
    def get_residual_analyser(self):
        residual_plot = residual_analysis.ResidualBaPlotter()
        residual_plot.add_data(self.forecaster, self.observed_rain)
        return residual_plot

    #implemented
    def get_coverage_analyser(self):
        year_segmentator = time_segmentation.YearSegmentator(
            self.forecaster.time_array)
        coverage = coverage_analysis.TimeSeries(year_segmentator)
        coverage.add_data(self.forecaster, self.observed_rain)
        return coverage

    #implemented
    def get_spread_analyser(self):
        all_inclusive = time_segmentation.AllInclusive(
            self.forecaster.time_array)
        coverage = coverage_analysis.TimeSeries(all_inclusive)
        coverage.credible_level_array = np.linspace(0.01, 0.99, 50)
        coverage.add_data(self.forecaster, self.observed_rain)
        return coverage

class Downscale(Printer):
    """For plotting forecast figures for TimeSeries

    Attributes:
        pool: object with the method map() for parallel computation
    """

    def __init__(self, forecaster, directory, pool, prefix=""):
        """
        Args:
            forecaster: compound_poisson.forecast.downscale_.Forecaster object
            directory: where to save the figures
            pool: object with the method map() for parallel computation
            prefix: what to put as a prefix for the file names for the figures
        """
        self.pool = pool
        super().__init__(forecaster, directory, prefix)

    #implemented
    def print_forecast(self):
        """Print figures for the forecasts

        The following figures are printed:
            -bias loss at each location (heat map), for both mean and median
            -median forecast, heat map for each day
            -for each location, everything in TimeSeries.print_forecast()
        """

        test_set = self.forecaster.data
        angle_resolution = dataset.ANGLE_RESOLUTION
        longitude_grid = test_set.topography["longitude"] - angle_resolution / 2
        latitude_grid = test_set.topography["latitude"] + angle_resolution / 2
        rain_units = test_set.rain_units

        series_dir = path.join(self.directory, "series_forecast")
        if not path.isdir(series_dir):
            os.mkdir(series_dir)
        map_dir = path.join(self.directory, "map_forecast")
        if not path.isdir(map_dir):
            os.mkdir(map_dir)

        #forecast map, 3 dimensions, same as test_set.rain
            #prediction of precipitation for each point in space and time
            #0th dimension is time, remaining is space
        forecast_map = ma.empty_like(test_set.rain)
        #array of dictionaries, one for each loss class
            #each element contains a dictionary of loss maps
        loss_map_array = []
        quantile_array = stats.norm.cdf([-1, 0, 1])
        for i_loss in range(len(loss_segmentation.LOSS_CLASSES)):
            dic = {}
            dic["bias_mean"] = ma.empty_like(test_set.rain[0])
            dic["bias_median"] = ma.empty_like(test_set.rain[0])
            loss_map_array.append(dic)

        #get forecast (median) and losses for the maps
        for forecaster_i, observed_rain_i in (
            zip(self.forecaster.generate_time_series_forecaster(),
                test_set.generate_unmask_rain())):
            lat_i = forecaster_i.time_series.id[0]
            long_i = forecaster_i.time_series.id[1]
            forecast_map[:, lat_i, long_i] = forecaster_i.forecast_median

            #get the value for each loss (to produce a map)
            for i_loss, Loss in enumerate(loss_segmentation.LOSS_CLASSES):
                loss_i = Loss(self.forecaster.n_simulation)
                loss_i.add_data(forecaster_i, observed_rain_i)

                loss_map_array[i_loss]["bias_mean"][lat_i, long_i] = (
                    loss_i.get_bias_loss())
                loss_map_array[i_loss]["bias_median"][lat_i, long_i] = (
                    loss_i.get_bias_median_loss())

            forecaster_i.del_memmap()

        #plot the losses map
        for i_loss, Loss in enumerate(loss_segmentation.LOSS_CLASSES):
            for metric, loss_map in loss_map_array[i_loss].items():
                plt.figure()
                ax = plt.axes(projection=crs.PlateCarree())
                im = ax.pcolor(longitude_grid,
                               latitude_grid,
                               loss_map,
                               cmap='Greys')
                ax.coastlines(resolution="50m")
                plt.colorbar(im)
                ax.set_aspect("auto", adjustable=None)
                plt.savefig(
                    path.join(self.directory,
                              self.prefix+Loss.get_short_name()+"_"+metric
                                +"_map.pdf"),
                    bbox_inches="tight")
                plt.close()

        #plot the spatial forecast for each time (in parallel)
        message_array = []
        for i, time in enumerate(test_set.time_array):
            title = "precipitation (" + rain_units + ") : " + str(time)
            file_path = path.join(map_dir, str(i) + ".png")
            message = PrintForecastMapMessage(forecast_map[i],
                                              latitude_grid,
                                              longitude_grid,
                                              title,
                                              file_path)
            message_array.append(message)
        self.pool.map(PrintForecastMapMessage.print, message_array)

        #plot the forecast (time series) for each location (in parallel)
        message_array = []
        for forecaster_i, observed_rain_i in (
            zip(self.forecaster.generate_forecaster_no_memmap(),
                test_set.generate_unmask_rain())):
            message = PrintForecastSeriesMessage(
                series_dir, forecaster_i, observed_rain_i)
            message_array.append(message)
        self.pool.map(PrintForecastSeriesMessage.print, message_array)

    #implemented
    def get_roc_curve_array(self, rain_warning_array, index=None):
        return self.forecaster.get_roc_curve_array(rain_warning_array, index)

    #implemented
    def get_distribution_comparer(self):
        return self.forecaster.compare_dist_with_observed()

    #implemented
    def get_loss_segmentator(self):
        return loss_segmentation.Downscale(self.forecaster)

    #implemented
    def get_residual_analyser(self):
        residual_plot = residual_analysis.ResidualBaPlotter()
        residual_plot.add_downscale(self.forecaster)
        return residual_plot

    #implemented
    def get_coverage_analyser(self):
        year_segmentator = time_segmentation.YearSegmentator(
            self.forecaster.time_array)
        coverage = coverage_analysis.Downscale(year_segmentator)
        coverage.add_data(self.forecaster)
        return coverage

    #implemented
    def get_spread_analyser(self):
        all_inclusive = time_segmentation.AllInclusive(
            self.forecaster.time_array)
        coverage = coverage_analysis.Downscale(all_inclusive)
        coverage.credible_level_array = np.linspace(0.01, 0.99, 50)
        coverage.add_data(self.forecaster)
        return coverage

class PrintForecastMapMessage(object):
    """For printing forecast over space for a given time point (in parallel)
    """

    def __init__(self,
                 forecast_i,
                 latitude_grid,
                 longitude_grid,
                 title,
                 file_path):
        self.forecast_i = forecast_i
        self.latitude_grid = latitude_grid
        self.longitude_grid = longitude_grid
        self.title = title
        self.file_path = file_path

    def print(self):
        #treat 0 mm as masked data
        self.forecast_i.mask[self.forecast_i == 0] = True

        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(self.longitude_grid,
                       self.latitude_grid,
                       self.forecast_i,
                       vmin=0,
                       vmax=15)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title(self.title)
        plt.savefig(self.file_path, bbox_inches="tight")
        plt.close()

class PrintForecastSeriesMessage(object):
    """For printing forecasts for each spatial point (in parallel)
    """

    def __init__(self, series_dir, forecaster, observed_rain_i):
        self.series_sub_dir =  path.join(
            series_dir, str(forecaster.time_series.id))
        self.forecaster = forecaster
        self.observed_rain_i = observed_rain_i
        if not path.exists(self.series_sub_dir):
            os.mkdir(self.series_sub_dir)

    def print(self):
        printer = TimeSeries(
            self.forecaster, self.observed_rain_i, self.series_sub_dir)
        #printer be responsible for load_memmap() and del_memmap() when print()
            #is called
        printer.print()
