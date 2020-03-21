import math

import numpy as np
from scipy import stats

from compound_poisson.mcmc import target

class TargetModelField(target.Target):

    def __init__(self, downscale, time_step):
        self.model_field_coarse = downscale.model_field_coarse
        self.model_field_shift = downscale.model_field_shift
        self.model_field_scale = downscale.model_field_scale
        self.prior_mean = None
        self.prior_cov_chol = None
        self.prior_scale = None

class TargetPrecision(target.Target):

    def __init__(self, downscale):
        self.precision_prior = get_precision_prior()
        self.precision = self.precision_prior.mean()
        self.precision_before = None

    def simulate_from_prior(self, rng):
        self.precision_prior.random_state = rng
        return np.asarray(self.precision_prior.rvs())

class TargetGp(target.Target):

    def __init__(self, downscale):
        self.model_field_target = downscale.model_field_target
        self.model_field_coarse = downscale.model_field_coarse
        self.topography_coarse = downscale.topography_coarse_normalise
        self.topography = downscale.topography_normalise
        self.gp_prior = get_gp_prior()
        self.gp_precision = self.gp_prior.mean()
        self.gp_precision_before = None
        self.mask = downscale.mask
        self.square_error_coarse = np.zeros(
            (downscale.n_coarse, downscale.n_coarse))
        self.square_error_fine = downscale.square_error
        self.square_error_coarse_fine = np.zeros(
            (downscale.n_coarse, downscale.area_unmask))

        n_coarse = downscale.n_coarse

        for topo_i in self.topography_coarse.values():
            topo_i = topo_i.flatten()
            for i in range(n_coarse):
                for j in range(i+1, n_coarse):
                    self.square_error_coarse[i, j] += math.pow(
                        topo_i[i] - topo_i[j], 2)
                    self.square_error_coarse[j, i] = (
                        self.square_error_coarse[i,j])

        unmask = np.logical_not(self.mask).flatten()
        for key, topo_fine in self.topography.items():
            topo_coarse = self.topography_coarse[key].flatten()
            topo_fine = topo_fine.flatten()
            topo_fine = topo_fine[unmask]
            for i in range(n_coarse):
                for j in range(downscale.area_unmask):
                    self.square_error_coarse_fine[i, j] += math.pow(
                        topo_coarse[i] - topo_fine[j], 2)

    def simulate_from_prior(self, rng):
        self.gp_prior.random_state = rng
        return np.asarray(self.gp_prior.rvs())


def get_precision_prior():
    prior = stats.gamma(a=0.5, scale=5)
    return prior

def get_gp_prior():
    return stats.gamma(a=0.5, scale=5)
