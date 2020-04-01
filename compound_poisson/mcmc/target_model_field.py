import math

import numpy as np
from numpy import linalg
from scipy import stats

from compound_poisson.mcmc import target

class TargetModelField(target.Target):

    def __init__(self, downscale, time_step):
        self.downscale = downscale
        self.time_step = time_step
        self.prior_mean = None
        self.model_field_gp_target = downscale.model_field_gp_target
        self.model_field_shift = []
        self.model_field_scale = []
        self.area_unmask = downscale.area_unmask
        self.n_model_field = downscale.n_model_field
        self.n_total_parameter = self.area_unmask * self.n_model_field

        self.prior_mean = np.empty(self.n_total_parameter)
        for i, model_field_coarse in enumerate(
            downscale.model_field_coarse.values()):
            self.model_field_shift.append(
                np.mean(model_field_coarse[time_step]))
            self.model_field_scale.append(
                np.std(model_field_coarse[time_step], ddof=1))
            self.prior_mean[i*self.area_unmask:(i+1)*self.area_unmask] = (
                self.model_field_shift[i])

    def get_n_dim(self):
        return self.n_total_parameter

    def get_state(self):
        return self.downscale.get_model_field(self.time_step)

    def update_state(self, state):
        self.downscale.set_model_field(state, self.time_step)
        self.downscale.update_all_cp_parameters()

    def get_log_prior(self):
        z = self.get_state()
        z -= self.prior_mean
        ln_prior_term = []
        chol = self.model_field_gp_target.cov_chol
        precision = self.model_field_gp_target.state["precision"]
        for i in range(self.n_model_field):
            z_i = z[i*self.area_unmask : (i+1)*self.area_unmask]
            z_i = linalg.lstsq(chol, z_i, rcond=None)[0]
            ln_prior_term.append(-0.5 * precision * np.dot(z_i, z_i))
        ln_det_cov = (2*np.sum(np.log(np.diagonal(chol))
            - self.area_unmask * math.log(precision)))
        return np.sum(ln_prior_term) - 0.5 * self.n_model_field * ln_det_cov

    def update_mean(self):
        gp_target = self.downscale.model_field_gp_target
        for i in range(self.n_model_field):
            self.prior_mean[i*self.area_unmask:(i+1)*self.area_unmask] = (
                gp_target.get_mean(i,
                                   self.time_step,
                                   self.model_field_shift[i],
                                   self.model_field_scale[i]))

    def simulate_from_prior(self, rng):
        model_field_vector = np.empty(self.area_unmask * self.n_model_field)
        scale = 1/math.sqrt(self.model_field_gp_target.state["precision"])
        cov_chol = self.model_field_gp_target.cov_chol
        for i in range(self.n_model_field):
            model_field_i = np.asarray(rng.normal(size=self.area_unmask))
            model_field_i = np.matmul(cov_chol, model_field_i)
            model_field_i *= self.model_field_scale[i]
            model_field_vector[i*self.area_unmask : (i+1)*self.area_unmask] = (
                model_field_i)
        model_field_vector *= scale
        model_field_vector += self.prior_mean
        return model_field_vector

class TargetModelFieldArray(target.Target):

    def __init__(self, downscale):
        self.downscale = downscale
        self.model_field_target_array = []
        self.area_unmask = downscale.area_unmask
        self.n_model_field = downscale.n_model_field
        self.n_parameter_i = self.area_unmask * self.n_model_field

        for i in range(len(downscale)):
            self.model_field_target_array.append(TargetModelField(downscale, i))

    def get_n_dim(self):
        n_parameter = 0
        for target in self:
            n_parameter += target.get_n_dim()
        return n_total_parameter

    def get_state(self):
        state = []
        for target in self:
            state.append(target.get_state())
        state = np.concatenate(state)
        return state

    def update_state(self, state):
        for time_step, target in enumerate(self):
            state_i = state[i*self.n_parameter_i : (i+1)*self.n_parameter_i]
            self.downscale.set_model_field(state_i, time_step)
        self.downscale.update_all_cp_parameters()

    def get_log_prior(self):
        ln_l = []
        for target in self:
            ln_l.append(target.get_log_prior())
        ln_l = np.sum(ln_l)
        return ln_l

    def update_mean(self):
        for target in self:
            target.update_mean()

    def simulate_from_prior(self, rng):
        state = []
        for target in self:
            state.append(target.simulate_from_prior(rng))
        state = np.concatenate(state)
        return state

    def __iter__(self):
        return iter(self.model_field_target_array)

    def __len__(self):
        return len(self.model_field_target_array)

    def __getitem__(self, index):
        return self.model_field_target_array[index]

    def __setitem__(self, index, value):
        self.model_field_target_array[index] = value

class TargetGp(target.Target):

    def __init__(self, downscale):
        self.downscale = downscale
        self.prior = {
            "precision": get_precision_prior(),
            "gp_precision": target.get_gp_precision_prior(),
            "reg_precision": get_regulariser_precision_prior(),
        }
        self.state = {}
        self.state_before = None

        self.model_field_coarse = downscale.model_field_coarse
        self.topography_coarse = downscale.topography_coarse_normalise
        self.topography = downscale.topography_normalise

        self.mask = downscale.mask
        self.square_error_coarse = np.zeros(
            (downscale.n_coarse, downscale.n_coarse))
        self.square_error_fine = downscale.square_error
        self.square_error_coarse_fine = np.empty(
            (downscale.n_coarse, downscale.area_unmask))
        self.cov_chol = None

        for key, prior in self.prior.items():
            self.state[key] = prior.mean()

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

    def get_n_dim(self):
        return len(self.prior)

    def get_state(self):
        return np.asarray(list(self.state))

    def update_state(self, state):
        for i, key in enumerate(self.state):
            self.state[key] = state[i]

    def get_log_likelihood(self):
        return self.downscale.get_model_field_log_pdf()

    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        ln_pdf = []
        for key, prior in self.prior.items():
            ln_pdf.append(prior.logpdf(self.state[key]))
        return np.sum(ln_pdf)

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

    def get_kernel_matrix(self, square_error):
        kernel_matrix = square_error.copy()
        kernel_matrix *= -self.state["gp_precision"] / 2
        kernel_matrix = np.exp(kernel_matrix)
        return kernel_matrix

    def get_mean(self, model_field_index, time_step, shift, scale):
        k_11 = self.get_kernel_matrix(self.square_error_coarse)
        self.regularise_kernel(k_11)
        model_field_name = list(self.model_field_coarse.keys())
        model_field_name = model_field_name[model_field_index]
        model_field = self.model_field_coarse[model_field_name][time_step]
        model_field = model_field.flatten()
        model_field -= shift
        model_field /= scale
        k_12 = self.get_kernel_matrix(self.square_error_coarse_fine)
        mean = np.matmul(k_12.T, linalg.lstsq(k_11, model_field, rcond=None)[0])
        mean *= scale
        mean += shift
        return mean

    def save_cov_chol(self):
        k_11_chol = self.get_kernel_matrix(self.square_error_coarse)
        self.regularise_kernel(k_11_chol)
        k_11_chol = linalg.cholesky(k_11_chol)
        k_12 = self.get_kernel_matrix(self.square_error_coarse_fine)
        cov_chol = linalg.lstsq(k_11_chol, k_12, rcond=None)[0]
        cov_chol = np.matmul(cov_chol.T, cov_chol)
        cov_chol = self.get_kernel_matrix(self.square_error_fine) - cov_chol
        cov_chol = linalg.cholesky(cov_chol)
        self.cov_chol = cov_chol

    def regularise_kernel(self, kernel):
        regulariser = 1/math.sqrt(self.state["reg_precision"])
        for i in range(kernel.shape[0]):
            kernel[i, i] += regulariser

def get_precision_prior():
    return stats.gamma(a=4/3, scale=3)

def get_regulariser_precision_prior():
    return stats.gamma(a=2, scale=1)
