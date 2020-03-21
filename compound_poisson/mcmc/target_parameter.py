import math

import numpy as np
from numpy import linalg
from scipy import stats

from compound_poisson.mcmc import target

class TargetParameter(target.Target):

    def __init__(self, downscale):
        super().__init__()
        self.downscale = downscale
        self.n_parameter = downscale.n_parameter
        self.n_total_parameter = downscale.n_total_parameter
        self.area_unmask = downscale.area_unmask
        self.prior_mean = None
        self.prior_cov_chol = None
        self.prior_scale_parameter = target.get_parameter_std_prior()
        self.prior_scale_arma = target.get_parameter_std_prior()
        self.parameter_before = None
        self.arma_index = None

        parameter_name_array = self.downscale.get_parameter_vector_name()
        self.prior_mean = target.get_parameter_mean_prior(parameter_name_array)
        self.prior_cov_chol = np.identity(self.area_unmask)
        self.arma_index = target.get_arma_index(parameter_name_array)

    def get_n_dim(self):
        return self.n_total_parameter

    def get_state(self):
        return self.downscale.get_parameter_vector()

    def update_state(self, state):
        self.downscale.set_parameter_vector(state)
        self.downscale.update_all_cp_parameters()

    def get_log_likelihood(self):
        return self.downscale.get_log_likelihood()

    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        z = self.get_state()
        z -= self.prior_mean

        ln_prior_term = []

        for i in range(self.n_parameter):
            z_i = z[i*self.area_unmask : (i+1)*self.area_unmask]
            chol = self.prior_cov_chol
            if self.arma_index[i*self.area_unmask]:
                z_i = linalg.lstsq(self.prior_scale_arma * chol, z_i)[0]
            else:
                z_i = linalg.lstsq(self.prior_scale_parameter * chol, z_i)[0]
            ln_prior_term.append(-0.5 * np.dot(z_i, z_i))

        return (-np.sum(np.log(np.diagonal(self.prior_cov_chol)))
            + np.sum(ln_prior_term))

    def save_state(self):
        self.parameter_before = self.get_state()

    def revert_state(self):
        self.update_state(self.parameter_before)

    def simulate_from_prior(self, rng):
        parameter_vector = self.prior_mean.copy()
        for i in range(self.n_parameter):
            parameter_i = np.asarray(rng.normal(size=self.area_unmask))
            chol = self.prior_cov_chol.copy()
            if self.arma_index[i*self.area_unmask]:
                chol *= self.prior_scale_arma
            else:
                chol *= self.prior_scale_parameter
            parameter_i = np.matmul(chol, parameter_i)
            parameter_vector[i*self.area_unmask : (i+1)*self.area_unmask] += (
                parameter_i)
        return parameter_vector.flatten()

class TargetPrecision(target.Target):

    def __init__(self, parameter_target):
        super().__init__()
        self.parameter_target = parameter_target
        #reg then arma
        self.prior = target.get_precision_prior()
        self.precision = []
        for prior in self.prior:
            self.precision.append(prior.mean())
        self.precision = np.asarray(self.precision)
        self.precision_before = None

    def get_n_dim(self):
        return len(self.precision)

    def get_state(self):
        return self.precision

    def update_state(self, state):
        self.precision = state.copy()

    def get_log_likelihood(self):
        return self.parameter_target.get_log_prior()

    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        ln_prior = 0
        for i, prior in enumerate(self.prior):
            ln_prior += prior.logpdf(self.precision[i])
        return ln_prior

    def save_state(self):
        self.precision_before = self.precision.copy()

    def revert_state(self):
        self.precision = self.precision_before

    def simulate_from_prior(self, rng):
        prior_simulate = []
        for prior in self.prior:
            prior.random_state = rng
            prior_simulate.append(prior.rvs())
        return np.asarray(prior_simulate)

class TargetGp(target.Target):

    def __init__(self, downscale):
        self.parameter_target = downscale.parameter_target
        self.prior_precision = get_gp_precision_prior()
        self.precision = np.asarray([self.prior_precision.mean()])
        self.precision_before = None
        self.square_error = downscale.square_error

    def get_n_dim(self):
        return 1

    def get_state(self):
        return self.precision

    def update_state(self, state):
        self.precision = state.copy()

    def get_log_likelihood(self):
        return self.parameter_target.get_log_prior()

    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        return self.prior_precision.logpdf(self.precision)[0]

    def save_state(self):
        self.precision_before = self.precision.copy()

    def revert_state(self):
        self.precision = self.precision_before

    def simulate_from_prior(self, rng):
        self.prior_precision.random_state = rng
        return self.prior_precision.rvs()

    def get_cov_chol(self):
        cov_chol = self.square_error.copy()
        cov_chol *= - self.precision / 2
        cov_chol = np.exp(cov_chol)
        cov_chol = linalg.cholesky(cov_chol)
        return cov_chol

def get_gp_precision_prior():
    return stats.gamma(a=0.72, loc=2.27, scale=8.1)
