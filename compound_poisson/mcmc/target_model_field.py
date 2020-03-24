import math

import numpy as np
from numpy import linalg
from scipy import stats

from compound_poisson.mcmc import target

class TargetModelField(target.Target):

    def __init__(self, downscale, time_step):
        self.time_step = time_step
        self.prior_mean = None
        self.prior_cov_chol = None
        self.model_field_shift = downscale.model_field_shift
        self.model_field_scale = downscale.model_field_scale
        self.area_unmask = downscale.area_unmask
        self.n_model_field = len(self.model_field_shift)
        self.prior_mean = np.empty(self.area_unmask * self.n_model_field)
        for i in range(self.n_model_field):
            self.prior_mean[i*self.area_unmask:(i+1)*self.area_unmask] = (
                self.model_field_shift[i])

    def update_mean(self, gp_target):
        for i in range(self.n_model_field):
            self.prior_mean[i*self.area_unmask:(i+1)*self.area_unmask] = (
                gp_target.get_mean(i, self.time_step))

    def simulate_from_prior(self, rng):
        #model_field_vector = np.empty(self.area_unmask * self.n_model_field)
        model_field_vector = np.zeros(self.area_unmask * self.n_model_field)

        #for i in range(self.n_model_field):
            #model_field_i = np.asarray(rng.normal(size=self.area_unmask))
            #model_field_i = np.matmul(self.prior_cov_chol, model_field_i)
            #model_field_i *= self.model_field_scale[i]
            #model_field_vector[i*self.area_unmask : (i+1)*self.area_unmask] = (
                #model_field_i)
        model_field_vector += self.prior_mean
        return model_field_vector

class TargetRegPrecision(target.Target):

    def __init__(self, downscale):
        self.prior = get_regulariser_prior()
        self.precision = self.prior.mean()
        self.precision_before = None

    def simulate_from_prior(self, rng):
        self.prior.random_state = rng
        return np.asarray(self.prior.rvs())

class TargetGp(target.Target):

    def __init__(self, downscale):
        self.model_field_coarse = downscale.model_field_coarse
        self.model_field_shift = downscale.model_field_shift
        self.model_field_scale = downscale.model_field_scale
        self.topography_coarse = downscale.topography_coarse_normalise
        self.topography = downscale.topography_normalise
        self.gp_regulariser = get_regulariser_prior().mean()
        self.gp_prior = get_gp_prior()
        self.gp_precision = self.gp_prior.mean()
        self.gp_precision_before = None
        self.mask = downscale.mask
        self.square_error_coarse = np.zeros(
            (downscale.n_coarse, downscale.n_coarse))
        self.square_error_fine = downscale.square_error
        self.square_error_coarse_fine = np.empty(
            (downscale.n_coarse, downscale.area_unmask))
        self.cov_chol = None

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

    def get_kernel_matrix(self, square_error):
        kernel_matrix = square_error.copy()
        kernel_matrix *= -self.gp_precision / 2
        kernel_matrix = np.exp(kernel_matrix)
        return kernel_matrix

    def get_mean(self, model_field_index, time_step):
        k_11 = self.get_kernel_matrix(self.square_error_coarse)
        self.regularise_kernel(k_11)
        model_field_name = list(self.model_field_coarse.keys())
        model_field_name = model_field_name[model_field_index]
        model_field = self.model_field_coarse[model_field_name][time_step]
        model_field = model_field.flatten()
        model_field -= self.model_field_shift[model_field_index]
        model_field /= self.model_field_scale[model_field_index]
        k_12 = self.get_kernel_matrix(self.square_error_coarse_fine)
        mean = np.matmul(k_12.T, linalg.lstsq(k_11, model_field)[0])
        mean *= self.model_field_scale[model_field_index]
        mean += self.model_field_shift[model_field_index]
        return mean

    def save_cov_chol(self):
        k_11_chol = self.get_kernel_matrix(self.square_error_coarse)
        self.regularise_kernel(k_11_chol)
        k_11_chol = linalg.cholesky(k_11_chol)
        k_12 = self.get_kernel_matrix(self.square_error_coarse_fine)
        cov_chol = linalg.lstsq(k_11_chol, k_12)[0]
        cov_chol = np.matmul(cov_chol.T, cov_chol)
        cov_chol = self.get_kernel_matrix(self.square_error_fine) - cov_chol
        cov_chol = linalg.cholesky(cov_chol)
        self.cov_chol = cov_chol

    def regularise_kernel(self, kernel):
        for i in range(kernel.shape[0]):
            kernel[i, i] += self.gp_regulariser

def get_regulariser_prior():
    prior = stats.gamma(a=1, scale=0.1)
    return prior

def get_gp_prior():
    return stats.gamma(a=0.01, loc=2.3, scale=100)
