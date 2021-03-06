import os
from os import path

from cartopy import crs
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from numpy import ma
from numpy import random

import compound_poisson
import dataset

class PriorSimulator(object):

    def __init__(self, figure_directory, seed):
        self.figure_directory = figure_directory
        self.n_simulate = 10
        self.downscale = None
        self.angle_resolution = dataset.ANGLE_RESOLUTION
        self.rng = random.RandomState(random.MT19937(seed.spawn(1)[0]))
        self.instantiate_downscale()
        self.downscale.set_rng(seed)

    def instantiate_downscale(self):
        self.downscale = compound_poisson.Downscale(
            dataset.AnaDual10Training(), (5, 5))

    def simulate(self, gp_precision=None):
        downscale = self.downscale
        gp_target = downscale.parameter_gp_target
        if not gp_precision is None:
            gp_target.update_state([gp_precision])
        else:
            is_reject = True
            while is_reject:
                try:
                    gp_target.set_from_prior(self.rng)
                    is_reject = False
                except(linalg.LinAlgError):
                    print("Reject!")
        #cannot use downscale.parameter_target.set_from_prior as this would
            #require a cp parameter update for all time steps which can cause
            #numerical problems
        downscale.parameter_log_precision_target.set_from_prior(self.rng)
        parameter = downscale.parameter_target.simulate_from_prior(self.rng)
        downscale.set_parameter_vector(parameter)
        downscale.simulate_i(0)

    def print_map(self, data, name, directory, clim=None):
        if clim is not None:
            vmin = clim[0]
            vmax = clim[1]
        else:
            vmin = None
            vmax = None
        latitude_grid = self.downscale.topography["latitude"]
        longitude_grid = self.downscale.topography["longitude"]
        plt.figure()
        ax = plt.axes(projection=crs.PlateCarree())
        im = ax.pcolor(longitude_grid - self.angle_resolution / 2,
                       latitude_grid + self.angle_resolution / 2,
                       data,
                       vmin=vmin,
                       vmax=vmax)
        ax.coastlines(resolution="50m")
        plt.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plt.title(name)
        plt.savefig(path.join(directory, name + ".pdf"))
        plt.close()

    def print(self, figure_directory=None, precision=None):

        if figure_directory is None:
            figure_directory = self.figure_directory
        if not path.isdir(figure_directory):
            os.mkdir(figure_directory)

        downscale = self.downscale
        try:
            for i_simulate in range(self.n_simulate):
                self.simulate(precision)
                poisson_rate = ma.empty(downscale.shape)
                gamma_mean = ma.empty(downscale.shape)
                rain = ma.empty(downscale.shape)
                for lat_i in range(downscale.shape[0]):
                    for long_i in range(downscale.shape[1]):
                        time_series = downscale.time_series_array[lat_i][long_i]
                        poisson_rate[lat_i, long_i] = np.exp(
                            time_series.poisson_rate["const"])
                        gamma_mean[lat_i, long_i] = np.exp(
                            time_series.gamma_mean["const"])
                        rain[lat_i, long_i] = time_series[0]
                poisson_rate.mask = downscale.mask
                gamma_mean.mask = downscale.mask
                rain.mask = downscale.mask
                rain.mask[rain==0] = True

                self.print_map(poisson_rate,
                               str(i_simulate) + "_poisson_rate",
                               figure_directory)
                self.print_map(gamma_mean,
                               str(i_simulate) + "_gamma_mean",
                               figure_directory)
                self.print_map(rain,
                               str(i_simulate) + "_rain",
                               figure_directory,
                               [0, 50])

                cov_chol = self.downscale.parameter_gp_target.cov_chol
                cov = np.dot(cov_chol, np.transpose(cov_chol))
                plt.figure()
                plt.imshow(cov)
                plt.colorbar()
                plt.savefig(
                    path.join(
                        figure_directory, str(i_simulate) + "_autocorr.pdf"))
                plt.close()
        except(linalg.LinAlgError):
            print(precision, "fail")

    def __call__(self):
        self.print()

class PriorGpSimulator(PriorSimulator):

    def __init__(self, figure_directory, rng):
        super().__init__(figure_directory, rng)

    def __call__(self):
        precision_array = np.linspace(278, 1000, 10)
        for i, precision in enumerate(precision_array):
            figure_directory_i = os.path.join(self.figure_directory, str(i))
            self.print(figure_directory_i, precision)
            file = open(os.path.join(figure_directory_i, "precision.txt"), "w")
            file.write(str(precision))
            file.close()
