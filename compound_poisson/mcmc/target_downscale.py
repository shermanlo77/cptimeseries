"""Implementations of Target for Downscale

Target
    <- TargetParameter
    <- TargetGp

Downscale
    <>1..*- Target (one for each component to sample)

Target
    <>1- Downscale
This allows the communication between Target distributions owned by
    Downscale

Mcmc
    <>1- Target
"""

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

    #implemented
    def get_n_dim(self):
        return self.n_total_parameter

    #implemented
    def get_state(self):
        return self.downscale.get_parameter_vector()

    #implemented
    def update_state(self, state):
        #set all time series parameters and update them so that the likelihood
            #can be evaluated
        self.downscale.set_parameter_vector(state)
        self.downscale.update_all_cp_parameters()

    #implemented
    def get_log_likelihood(self):
        #a self.downscale.update_all_cp_parameters() is required beforehand
        return self.downscale.get_log_likelihood()

    #implemented
    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        #required when doing mcmc on the gp precision parameters
        #evaluate the multivariate Normal distribution
        z = self.get_state()
        z -= self.prior_mean
        ln_prior_term = []
        #retrive covariance information from parameter_gp_target
        gp_target = self.downscale.parameter_gp_target
        log_precision_target = self.downscale.parameter_log_precision_target
        chol = gp_target.cov_chol

        std_reg = np.exp(-0.5*log_precision_target.get_reg_state())
        std_arma = np.exp(-0.5*log_precision_target.get_arma_state())

        chol_reg = chol.copy()
        chol_arma = chol.copy()
        ln_det_cov_reg = 2*np.sum(np.log(np.diagonal(chol_reg)))
        ln_det_cov_arma = 2*np.sum(np.log(np.diagonal(chol_arma)))

        for i in range(len(chol)):
            chol_reg[i] *= std_reg[i]
            chol_arma[i] *= std_arma[i]

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

            chol_i = None
            ln_det_cov_i = None
            if self.arma_index[i*self.area_unmask]:
                chol_i = chol_arma
                ln_det_cov_i = ln_det_cov_arma
            else:
                chol_i = chol_reg
                ln_det_cov_i = ln_det_cov_reg
            #reminder: det(cX) = c^d det(X)
            #reminder: det(L*L) = det(L) * det(L)
            #reminder: 1/sqrt(precision) = standard deviation or scale
            z_i = linalg.solve_triangular(chol_i, z_i, lower=True)
            ln_prior_term.append(-0.5 * (np.dot(z_i, z_i) + ln_det_cov_i))

        return np.sum(ln_prior_term)

    #implemented
    def save_state(self):
        self.parameter_before = self.get_state()

    #implemented
    def revert_state(self):
        self.update_state(self.parameter_before)

    #implemented
    def simulate_from_prior(self, rng):
        #required for study of prior distribution and elliptical slice sampling
        parameter_vector = self.prior_mean.copy()
        log_precision_target = self.downscale.parameter_log_precision_target
        #cholesky of kernel matrix
        chol = self.downscale.parameter_gp_target.cov_chol
        #simulate each parameter, correlation only in space, not between
            #parameters
        std_reg = np.exp(-0.5*log_precision_target.get_reg_state())
        std_arma = np.exp(-0.5*log_precision_target.get_arma_state())
        for i in range(self.n_parameter):
            if self.arma_index[i*self.area_unmask]:
                std_array = std_arma
            else:
                std_array = std_reg
            chol_i = chol.copy()
            for i_row in range(len(chol_i)):
                chol_i[i_row] *= std_array[i_row]

            parameter_i = np.asarray(rng.normal(size=self.area_unmask))
            parameter_i = np.matmul(chol_i, parameter_i)
            parameter_vector[i*self.area_unmask:(i+1)*self.area_unmask] += (
                parameter_i)

        return parameter_vector

    def get_prior_mean(self):
        #implemented
        return self.prior_mean

class TargetLogPrecision(target.Target):
    """Target implementation for the log precision for the parameters

    Each location has 2 log precisions, one for regression and constant terms,
        the other for ARMA.

    Attributes:
        downscale: pointer to parent downscale object
        area_unmask: number of spatial points on land
        prior: dictionary of distributions
        state: numpy array of log precisions
        state_before: deep copy of state after calling save_state()
    """

    def __init__(self, downscale):
        self.downscale = downscale
        self.area_unmask = self.downscale.area_unmask
        self.prior = target.get_log_precision_prior()
        self.state = None
        self.state_before = None

        self.state = []
        for prior in self.prior.values():
            for i_location in range(self.area_unmask):
                self.state.append(prior.mean())
        self.state = np.asarray(self.state)

    #implemented
    def get_n_dim(self):
        return self.downscale.area_unmask * 2

    #implemented
    def get_state(self):
        return self.state

    def get_sub_state(self, prior_key):
        """Extract the vector of log precisions for a parameter
        """
        for i, key in enumerate(self.prior):
            if key == prior_key:
                return self.state[i*self.area_unmask : (i+1)*self.area_unmask]

    def get_reg_state(self):
        """Return vector of log precisions for regression parameters
        """
        return self.get_sub_state("log_precision_reg")

    def get_arma_state(self):
        """Return vector of log precisions for ARMA parameters
        """
        return self.get_sub_state("log_precision_arma")

    #implemented
    def update_state(self, state):
        self.state = state

    #implemented
    def get_log_likelihood(self):
        return self.downscale.parameter_target.get_log_prior()

    def get_log_prior(self):
        ln_prior = []
        for i_prior, prior in enumerate(self.prior.values()):
            for i_location in range(self.area_unmask):
                state_i = self.state[i_prior * self.area_unmask + i_location]
                ln_prior.append(prior.logpdf(state_i))
        return np.sum(ln_prior)

    #implemented
    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    #implemented
    def save_state(self):
        self.state_before = self.state.copy()

    #implemented
    def revert_state(self):
        self.state = self.state_before

    #implemented
    def simulate_from_prior(self, rng):
        state = []
        for prior in self.prior.values():
            prior.random_state = rng
            for i_location in range(self.area_unmask):
                state.append(prior.rvs())
        return np.asarray(state)

    #implemented
    def get_prior_mean(self):
        state = []
        for prior in self.prior.values():
            for i_location in range(self.area_unmask):
                state.append(prior.mean())
        return np.asarray(state)

class TargetGp(target.Target):
    """Contains density information for the GP precision parameters

    Attributes:
        downscale: pointer to parent Downscale object
        prior: scipy.stats prior distributions
        state: gp precision
        state_before: copy of state when doing mcmc (for reverting after
            rejection)
        cov_chol: kernel matrix, cholesky
        cov_chol_before: copy of cov_chol when doing mcmc (for reverting after
            rejection)
        square_error: matrix (area_unmask x area_unmask) containing square error
            of topography between each point in space.
    """

    def __init__(self, downscale):
        super().__init__()
        self.downscale = downscale
        #keys for prior and state are in order:
            #precision_reg, precision_arma, gp_precision
        self.prior = target.get_gp_precision_prior()
        self.state = self.prior.mean()
        self.state_before = None
        self.cov_chol = None
        self.cov_chol_before = None
        #square_error does not change so points to the copy owned by Downscale
        self.square_error = downscale.square_error

        self.state = self.prior.median()

    #implemented
    def get_n_dim(self):
        return 1

    #implemented
    def get_state(self):
        return np.asarray([self.state])

    #implemented
    def update_state(self, state):
        self.state = state[0]
        self.save_cov_chol()

    #implemented
    def get_log_likelihood(self):
        return self.downscale.parameter_target.get_log_prior()

    #implemented
    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        return self.prior.logpdf(self.state)

    #implemented
    def save_state(self):
        #make a deep copy of the state and the covariance matrix
        self.state_before = self.state
        self.cov_chol_before = self.cov_chol.copy()

    #implemented
    def revert_state(self):
        self.state = self.state_before
        self.cov_chol = self.cov_chol_before

    #implemented
    def simulate_from_prior(self, rng):
        self.prior.random_state = rng
        return np.asarray([self.prior.rvs()])

    def save_cov_chol(self):
        """Calculate and save the Cholesky decomposition of the kernel matrix
        """
        cov_chol = self.square_error.copy()
        cov_chol *= -self.state / 2
        cov_chol = np.exp(cov_chol)
        cov_chol = linalg.cholesky(cov_chol, True)
        self.cov_chol = cov_chol
