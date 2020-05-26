import math

import numpy as np
from scipy import linalg
from scipy import stats

from compound_poisson.mcmc import target

class TargetParameter(target.Target):
    """Target object for MCMC to sample the parameters for a collection of
        compound Poisson time-series

    Attributes:
        downscale: pointer to parent
        n_parameter: number of parameters for a time series (or location)
        n_total_parameter: number of parameters for all time series (for all
            locations on fine grid)
        area_unmask: number of points on fine grid
        prior_mean: mean vector of the GP
        parameter_before: place to save parameter when doing MCMC
        arma_index: area of boolean, true if it is ARMA term
    """

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
        #implemented
        return self.n_total_parameter

    def get_state(self):
        #implemented
        return self.downscale.get_parameter_vector()

    def update_state(self, state):
        #implemented
        #set all time series parameters and update them so that the likelihood
            #can be evaluated
        self.downscale.set_parameter_vector(state)
        self.downscale.update_all_cp_parameters()

    def get_log_likelihood(self):
        #implemented
        #a self.downscale.update_all_cp_parameters() is required beforehand
        return self.downscale.get_log_likelihood()

    def get_log_target(self):
        #implemented
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        #required when doing mcmc on the gp precision parameters
        #evaluate the multivariate Normal distribution
        z = self.get_state()
        z -= self.prior_mean
        ln_prior_term = []
        #retrive covariance information from parameter_gp_target
        gp_target = self.downscale.parameter_gp_target
        chol = gp_target.cov_chol
        #evaluate the Normal density for each parameter, each parameter has a
            #covariance matrix which represent the correlation in space. There
            #is no correlation between different parameters. In other words,
            #if a vector contains all parameters for all points in space, the
            #covariance matrix has a block diagonal structure. Evaluating
            #the density with a block digonal covariance matrix can be done
            #using a for loop
        for i in range(self.n_parameter):
            #z is the parameter - mean
            z_i = z[i*self.area_unmask : (i+1)*self.area_unmask]
            if self.arma_index[i*self.area_unmask]:
                precision = gp_target.state["precision_arma"]
            else:
                precision = gp_target.state["precision_reg"]
            z_i = linalg.solve_triangular(chol, z_i, lower=True)
            #reminder: det(cX) = c^d det(X)
            #reminder: det(L*L) = det(L) * det(L)
            #reminder: 1/sqrt(precision) = standard deviation or scale
            ln_det_cov = (2*np.sum(np.log(np.diagonal(chol)))
                - self.area_unmask * math.log(precision))
            ln_prior_term.append(
                -0.5 * (precision * np.dot(z_i, z_i) + ln_det_cov))
        return np.sum(ln_prior_term)

    def save_state(self):
        #implemented
        self.parameter_before = self.get_state()

    def revert_state(self):
        #implemneted
        self.update_state(self.parameter_before)

    def simulate_from_prior(self, rng):
        #implemented
        #required for study of prior distribution and slice sampling
        parameter_vector = self.prior_mean.copy()
        gp_target = self.downscale.parameter_gp_target
        chol = gp_target.cov_chol #cholesky of kernel matrix
        #simulate each parameter, correlation only in space, not between
            #parameters
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

    def get_prior_mean(self):
        #implemented
        return self.prior_mean

class TargetGp(target.Target):
    """Contains density information for the GP precision parameters

    Attributes:
        downscale: pointer to parent
        prior: dictionary of scipy.stats objects to represent the prior
            distributions
        state: dictionary of the different precision parameters
        state_before: copy of state when doing mcmc (ie reverting after
            rejection)
        cov_chol: kernel matrix, cholesky
        cov_chol_before: copy of cov_chol when doing mcmc (ie reverting after
            rejection)
        square_error: matrix (area_unmask x area_unmask) containing square error
            of topography between each point in space
    """

    def __init__(self, downscale):
        super().__init__()
        self.downscale = downscale
        #keys for prior and state are in order:
            #precision_reg, precision_arma, gp_precision
        self.prior = {}
        self.state = {}
        self.state_before = None
        self.cov_chol = None
        self.cov_chol_before = None
        self.square_error = downscale.square_error

        self.prior = target.get_precision_prior()
        self.prior["gp_precision"] = target.get_gp_precision_prior()
        #initalise using the mean of the prior distributions
        for key, prior in self.prior.items():
            self.state[key] = prior.mean()

    def get_n_dim(self):
        #implemneted
        return len(self.state)

    def get_state(self):
        #implemented
        return np.asarray(list(self.state.values()))

    def update_state(self, state):
        #implmented
        #update state and the covariance matrix
        for i, key in enumerate(self.prior):
            self.state[key] = state[i]
        self.save_cov_chol()

    def get_log_likelihood(self):
        #implemented
        return self.downscale.parameter_target.get_log_prior()

    def get_log_target(self):
        #implemented
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        ln_prior = 0
        for i, key in enumerate(self.prior):
            ln_prior += self.prior[key].logpdf(self.state[key])
        return ln_prior

    def save_state(self):
        #implemented
        self.state_before = self.state.copy()
        self.cov_chol_before = self.cov_chol.copy()

    def revert_state(self):
        #implemented
        self.state = self.state_before
        self.cov_chol = self.cov_chol_before

    def simulate_from_prior(self, rng):
        #implemented
        prior_simulate = []
        for prior in self.prior.values():
            prior.random_state = rng
            prior_simulate.append(prior.rvs())
        return np.asarray(prior_simulate)

    def save_cov_chol(self):
        """Calculate the kernel matrix
        """
        cov_chol = self.square_error.copy()
        cov_chol *= -self.state["gp_precision"] / 2
        cov_chol = np.exp(cov_chol)
        cov_chol = linalg.cholesky(cov_chol, True)
        self.cov_chol = cov_chol
