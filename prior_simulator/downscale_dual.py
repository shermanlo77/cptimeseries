import math
import os
from os import path

from cartopy import crs
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from numpy import ma
from scipy import stats

import prior_simulator
import compound_poisson
import dataset

class PriorSimulator(prior_simulator.downscale.PriorSimulator):

    def __init__(self, figure_directory, seed):
        super().__init__(figure_directory, seed)

    def instantiate_downscale(self):
        data = dataset.AnaDual1Training()
        data.pool = compound_poisson.multiprocess.Serial()
        self.downscale = compound_poisson.DownscaleDual(data, (5,5))

    def simulate(self, reg_precision=None):
        downscale = self.downscale
        model_field_gp_target = downscale.model_field_gp_target
        state_value = model_field_gp_target.simulate_from_prior(self.rng)
        for i, key in enumerate(model_field_gp_target.state):
            model_field_gp_target.state[key] = state_value[i]
        if not reg_precision is None:
            model_field_gp_target.state["regulariser_precision"] = reg_precision
        model_field_gp_target.save_cov_chol()

        #only interested in the first time point (to save on computation time)
        downscale.model_field_target.model_field_target_array = [
            downscale.model_field_target.model_field_target_array[0]]

        model_field = (downscale.model_field_target[0]
            .simulate_from_prior(self.rng))
        return model_field

    def print(self, figure_directory=None, reg_precision=None):

        if figure_directory is None:
            figure_directory = self.figure_directory
        if not path.isdir(figure_directory):
            os.mkdir(figure_directory)

        downscale = self.downscale
        for i_simulate in range(self.n_simulate):
            while True:
                try:
                    model_field = self.simulate(reg_precision)
                    for i_model_field, model_field_key in enumerate(
                        downscale.model_field_units):
                        #downscale.model_field_units is a dic, loop return keys
                        mask = downscale.mask
                        model_field_i = ma.empty(mask.shape)
                        model_field_i.mask = mask
                        model_field_i[np.logical_not(mask)] = model_field[
                            i_model_field*downscale.area_unmask
                            : (i_model_field+1)*downscale.area_unmask]
                        name = model_field_key + "_" + str(i_simulate)
                        self.print_map(model_field_i, name, figure_directory)
                    break
                except(linalg.LinAlgError):
                    print("Fail")
                    gp_target = downscale.model_field_gp_target
                    print("gp precision", gp_target.state["gp_precision"])
                    print("regulariser",
                          1/math.sqrt(gp_target.state["regulariser_precision"]))

    def __call__(self):
        self.print()

class PriorRegulariserSimulator(PriorSimulator):

    def __init__(self, figure_directory, rng):
        super().__init__(figure_directory, rng)

    def __call__(self):
        regulariser_array = np.power(10, np.linspace(-10, 10, 11))
        for i, regulariser in enumerate(regulariser_array):
            figure_directory_i = os.path.join(self.figure_directory, str(i))
            self.print(figure_directory_i, 1/math.pow(regulariser,2))
            file = open(
                os.path.join(figure_directory_i, "regulariser.txt"), "w")
            file.write(str(regulariser))
            file.close()
