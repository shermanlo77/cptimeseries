"""Implementations of Target for Downscale

Target
    <- TargetParameter
    <- TargetGp

Downscale
    <>1..*- Target (one for each component to sample)

Target
    <>1- Downscale
This allows the communication between Target distributions owned by MultiSeries
    or Downscale

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

    IMPORTANT: A new instance will not have its cov_chol_array initalised.
        You must call (or already implemented via some other class) the method
        update_cov_chol() when self.downscale.parameter_gp_target becomes
        available.

    Attributes:
        downscale: pointer to parent
        n_parameter: number of parameters for a time series (or location)
        n_total_parameter: number of parameters for all time series (for all
            locations on fine grid)
        area_unmask: number of points on fine grid
        prior_mean: mean vector of the GP
        parameter_before: place to save parameter when doing MCMC
        cov_chol_array: array of cov chol, dim 0: one for each parameter, dim 1
            and dim 2: cholesky of covariance matrix
    """

    def __init__(self, downscale):
        super().__init__()
        self.downscale = downscale
        self.n_parameter = downscale.n_parameter
        self.n_total_parameter = downscale.n_total_parameter
        self.area_unmask = downscale.area_unmask
        self.prior_mean = None
        self.parameter_before = None
        self.cov_chol_array = np.zeros(
            (self.n_parameter, self.area_unmask, self.area_unmask))

        parameter_name_array = self.downscale.get_parameter_vector_name()
        self.prior_mean = target.get_parameter_mean_prior(parameter_name_array)

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
        #assume each parameter are independent, the cholesky covariance matrix
            #for parameter i is self.cov_chol_array[i]
        for i in range(self.n_parameter):
            #z is the parameter - mean
            z_i = z[i*self.area_unmask : (i+1)*self.area_unmask]
            chol_i = self.cov_chol_array[i]

            #reminder: det(cX) = c^d det(X)
            #reminder: det(L*L) = det(L) * det(L)
            #reminder: 1/sqrt(precision) = standard deviation or scale
            ln_det_cov_term = -np.sum(np.log(np.diagonal(chol_i)))
            z_i = linalg.solve_triangular(chol_i, z_i, lower=True)
            ln_prior_term.append(ln_det_cov_term)
            ln_prior_term.append(-0.5 * np.dot(z_i, z_i))

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
        #required for elliptical slice sampling
        parameter_vector = self.get_prior_mean()
        #assume each parameter are independent, the cholesky covariance matrix
            #for parameter i is self.cov_chol_array[i]
        for i in range(self.n_parameter):
            chol_i = self.cov_chol_array[i]
            parameter_i = np.asarray(rng.normal(size=self.area_unmask))
            parameter_i = np.matmul(chol_i, parameter_i)
            parameter_vector[i*self.area_unmask:(i+1)*self.area_unmask] += (
                parameter_i)

        return parameter_vector

    def get_prior_mean(self):
        #implemented
        return self.prior_mean.copy()

    def update_cov_chol(self):
        """Update the covariance matrix

        Update the member variable cov_chol_array using Gaussian process
            parameters in TargetGp self.downscale.parameter_gp_target.
        """
        log_precision_target = self.downscale.parameter_log_precision_target
        gp_target = self.downscale.parameter_gp_target
        #update the covariance matrix for each parameter
        for i in range(self.n_parameter):

            #calculate the cholesky of the gram matrix (sometimes referred to as
                #the kernel matrix)
            cov_chol = self.downscale.square_error.copy()
            gp_precision = gp_target.precision_gp_state[i]
            cov_chol *= -gp_precision / 2
            cov_chol = np.exp(cov_chol)
            cov_chol = linalg.cholesky(cov_chol, True)
            #scale the cholesky matrix by a precision
            cov_chol *= np.exp(-0.5*log_precision_target.log_precision_state[i])
            self.cov_chol_array[i] = cov_chol

class TargetLogPrecision(target.Target):
    """Contains information and parameters for the precision scale of the
        Gaussian process prior

    Contains information and parameters for the precision scale of the Gaussian
        process prior (such as the density, prior and state vector). The
        gaussian process prior has the form N(0, exp(-log_precision_reg) * K)
        where K is the gram matrix. log_precision_reg is for the regression
        parameters, which can be replaced with log_precision_arma for the ARMA
        parameters. Each parameter has a different precision scale.

    Attributes:
        downscale: pointer to parent Downscale object
        log_precision_reg_prior: prior distribution for the log precision scale
            for the regression parameters
        log_precision_arma_prior: prior distribution for the log precision scale
            for the ARMA parameters
        arma_index: vector of boolean, True is this parameter is an ARMA
            parameter, else it is a regression parameter
        log_precision_state: numpy vector of precision values for each parameter
    """

    def __init__(self, downscale):
        super().__init__()
        self.downscale = downscale
        prior = target.get_log_precision_prior()
        self.log_precision_reg_prior = prior["log_precision_reg"]
        self.log_precision_arma_prior = prior["log_precision_arma"]
        parameter_name_array = self.downscale.get_parameter_vector_name()
        self.arma_index = target.get_arma_index(parameter_name_array)
        self.log_precision_state = self.get_prior_mean()

    #implemented
    def get_n_dim(self):
        return self.log_precision_state.size

    #implemented
    def get_state(self):
        return self.log_precision_state

    #implemented
    def update_state(self, state):
        self.log_precision_state = state
        self.update_cov_chol()

    #implemented
    def get_log_likelihood(self):
        return self.downscale.parameter_target.get_log_prior()

    #implemented
    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        log_pdf_terms = []
        for i_parameter in range(self.downscale.n_parameter):
            parameter_i = self.log_precision_state[i_parameter]
            if self.arma_index[i_parameter]:
                log_pdf_terms.append(
                    self.log_precision_arma_prior.logpdf(parameter_i))
            else:
                log_pdf_terms.append(
                    self.log_precision_reg_prior.logpdf(parameter_i))
        return np.asarray(log_pdf_terms)

    #implemented
    def get_prior_mean(self):
        mean = []
        for i_parameter in range(self.downscale.n_parameter):
            if self.arma_index[i_parameter]:
                mean.append(self.log_precision_arma_prior.mean())
            else:
                mean.append(self.log_precision_reg_prior.mean())
        return np.asarray(mean)

    #implemented
    def simulate_from_prior(self, rng):
        self.log_precision_reg_prior.random_state = rng
        self.log_precision_arma_prior.random_state = rng
        state_vector = []
        for i_parameter in range(self.downscale.n_parameter):
            parameter_i = self.log_precision_state[i_parameter]
            if self.arma_index[i_parameter]:
                state_vector.append(self.log_precision_arma_prior.rvs())
            else:
                state_vector.append(self.log_precision_reg_prior.rvs())
        return np.asarray(state_vector)

    def update_cov_chol(self):
        """Calculate and save the Cholesky decomposition of the covariance
            matrices
        """
        self.downscale.parameter_target.update_cov_chol()

class TargetGp(target.Target):
    """Contains information and parameters for the kernel parameter of the
        Gaussian process prior

    Contains information and parameters for the kernel parameter of Gaussian
        process prior (such as the density, prior and state vector). The
        gaussian process prior has the form N(0, exp(-precision_reg) * K) where
        K is the gram matrix. The kernel is exp(-0.5*precision_gp*square_error).
        precision_gp is the parameter for the Gaussian kernel. Each parameter
        has a different kernel parameter.

    Attributes:
        downscale: pointer to parent Downscale object
        state_before: numpy vector, used for temporarily storing the entire
            state vector
        cov_chol_array_before: numpy array, used for temporarily storing the
            covariance matrix
        precision_gp_prior: prior distribution for the Gaussian kernel parameter
        precision_gp_state: numpy vector, gaussian kernel parameter for each
            parameter
    """

    def __init__(self, downscale):
        super().__init__()
        self.downscale = downscale

        self.state_before = None
        self.cov_chol_array_before = None

        self.precision_gp_prior = target.get_gp_precision_prior()
        self.precision_gp_state = np.zeros(self.downscale.n_parameter)
        self.precision_gp_state[:] = self.precision_gp_prior.median()

    #implemented
    def get_n_dim(self):
        return self.precision_gp_state.size

    #implemented
    def get_state(self):
        return self.precision_gp_state

    #implemented
    def update_state(self, state):
        self.update_state_vector(state)
        self.update_cov_chol()

    def update_state_vector(self, state):
        self.precision_gp_state = state

    #implemented
    def get_log_likelihood(self):
        return self.downscale.parameter_target.get_log_prior()

    #implemented
    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        log_pdf_terms = self.precision_gp_prior.logpdf(self.precision_gp_state)
        return np.sum(log_pdf_terms)

    #implemented
    def save_state(self):
        #make a deep copy of the state and the covariance matrix
        self.state_before = self.get_state()
        self.cov_chol_array_before = (
            self.downscale.parameter_target.cov_chol_array.copy())

    #implemented
    def revert_state(self):
        self.update_state_vector(self.state_before)
        self.downscale.parameter_target.cov_chol_array = (
            self.cov_chol_array_before)

    #implemented
    def simulate_from_prior(self, rng):
        self.precision_gp_prior.random_state = rng
        downscale = self.downscale
        precision_gp = self.precision_gp_prior.rvs(downscale.n_parameter)
        return precision_gp

    def update_cov_chol(self):
        """Calculate and save the Cholesky decomposition of the covariance
            matrices
        """
        self.downscale.parameter_target.update_cov_chol()
