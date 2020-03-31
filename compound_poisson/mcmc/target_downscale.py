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
        self.parameter_before = None
        self.arma_index = None

        parameter_name_array = self.downscale.get_parameter_vector_name()
        self.prior_mean = target.get_parameter_mean_prior(parameter_name_array)
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
        gp_target = self.downscale.parameter_gp_target
        chol = gp_target.cov_chol
        for i in range(self.n_parameter):
            z_i = z[i*self.area_unmask : (i+1)*self.area_unmask]
            if self.arma_index[i*self.area_unmask]:
                precision = gp_target.state["precision_arma"]
            else:
                precision = gp_target.state["precision_reg"]
            z_i = linalg.lstsq(chol, z_i, rcond=None)[0]
            ln_det_cov = (2*np.sum(np.log(np.diagonal(chol)))
                - self.area_unmask * math.log(precision))
            ln_prior_term.append(
                -0.5 * (precision * np.dot(z_i, z_i) + ln_det_cov))
        return np.sum(ln_prior_term)

    def save_state(self):
        self.parameter_before = self.get_state()

    def revert_state(self):
        self.update_state(self.parameter_before)

    def simulate_from_prior(self, rng):
        parameter_vector = self.prior_mean.copy()
        gp_target = self.downscale.parameter_gp_target
        chol = gp_target.cov_chol
        for i in range(self.n_parameter):
            parameter_i = np.asarray(rng.normal(size=self.area_unmask))
            parameter_i = np.matmul(chol, parameter_i)
            if self.arma_index[i*self.area_unmask]:
                precision = gp_target.state["precision_arma"]
            else:
                precision = gp_target.state["precision_reg"]
            parameter_i *= 1/math.sqrt(precision)
            parameter_vector[i*self.area_unmask : (i+1)*self.area_unmask] += (
                parameter_i)
        return parameter_vector

class TargetGp(target.Target):

    def __init__(self, downscale):
        super().__init__()
        self.parameter_target = downscale.parameter_target
        self.prior = {}
        self.state = {}
        self.square_error = downscale.square_error
        self.cov_chol = None

        self.prior = target.get_precision_prior()
        self.prior["gp_precision"] = target.get_gp_precision_prior()
        for key, prior in self.prior.items():
            self.state[key] = prior.mean()

        self.state_before = None

    def get_n_dim(self):
        return len(self.state)

    def get_state(self):
        return np.asarray(list(self.state.values()))

    def update_state(self, state):
        for i, key in enumerate(self.prior):
            self.state[key] = state[i]

    def get_log_likelihood(self):
        return self.parameter_target.get_log_prior()

    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        ln_prior = 0
        for i, key in enumerate(self.prior):
            ln_prior += self.prior[key].logpdf(self.state[key])
        return ln_prior

    def save_state(self):
        self.state_before = self.state.copy()

    def revert_state(self):
        self.state = self.state_before

    def simulate_from_prior(self, rng):
        prior_simulate = []
        for prior in self.prior.values():
            prior.random_state = rng
            prior_simulate.append(prior.rvs())
        return np.asarray(prior_simulate)

    def save_cov_chol(self):
        cov_chol = self.square_error.copy()
        cov_chol *= -self.state["gp_precision"] / 2
        cov_chol = np.exp(cov_chol)
        cov_chol = linalg.cholesky(cov_chol)
        self.cov_chol = cov_chol
