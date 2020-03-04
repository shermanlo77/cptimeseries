import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plot
import numpy as np
from numpy.linalg import LinAlgError

import compound_poisson as cp
from dataset import Ana_1, Data

class PriorDsSimulator:
    
    def __init__(self, figure_directory, rng):
        self.figure_directory = figure_directory
        self.n_simulate = 10
        self.downscale = cp.Downscale(Ana_1(), (5, 5))
        self.downscale.set_rng(rng)
        self.angle_resolution = Data.ANGLE_RESOLUTION
        self.rng = rng
        if not os.path.isdir(figure_directory):
            os.mkdir(figure_directory)
    
    def simulate(self, precision=None):
        downscale = self.downscale
        downscale.set_parameter_from_prior()
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
        plot.figure()
        ax = plot.axes(projection=ccrs.PlateCarree())
        im = ax.pcolor(longitude_grid - self.angle_resolution / 2,
                       latitude_grid + self.angle_resolution / 2,
                       data,
                       vmin=vmin,
                       vmax=vmax)
        ax.coastlines(resolution="50m")
        plot.colorbar(im)
        ax.set_aspect("auto", adjustable=None)
        plot.title(name)
        plot.savefig(os.path.join(directory, name + ".pdf"))
        plot.close()
    
    def print(self, figure_directory=None, precision=None):
        
        if figure_directory is None:
            figure_directory = self.figure_directory
        if not os.path.isdir(figure_directory):
            os.mkdir(figure_directory)
        
        downscale = self.downscale
        try:
            for i_simulate in range(self.n_simulate):
                self.simulate(precision)
                poisson_rate = np.ma.empty(downscale.shape)
                gamma_mean = np.ma.empty(downscale.shape)
                rain = np.ma.empty(downscale.shape)
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
                
                cov_chol = self.downscale.parameter_target.prior_cov_chol
                cov = np.dot(cov_chol, np.transpose(cov_chol))
                plot.figure()
                plot.imshow(cov)
                plot.colorbar()
                plot.savefig(
                    os.path.join(
                        figure_directory, str(i_simulate) + "_autocorr.pdf"))
                plot.close()
        except(LinAlgError):
            print(precision, "fail")
    
    def __call__(self):
        self.print()
