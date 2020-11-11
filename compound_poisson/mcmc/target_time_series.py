"""Implementations of Target for TimeSeries

Target
    <- TargetParameter
    <- TargetZ
    <- TargetPrecision

TimeSeriesMcmc
    <>1..*- Target (one for each component to sample)

target.Target
    <>1- TimeSeriesMcmc
This allows the communication between Target distributions owned by
    TimeSeriesMcmc

Mcmc
    <>1- Target
"""

import numpy as np
from scipy import stats

from compound_poisson.mcmc import target

class TargetParameter(target.Target):
    """Contains the posterior distribution of the beta parameters

    Attributes:
        time_series: TimeSeries object being wrapped around
        prior_mean: prior mean vector
        prior_cov_chol: prior std vector
        cp_parameter_before: copy of time_series.cp_parameter_array when
            save_state() is called
    """

    def __init__(self, time_series):
        """
        Args:
            time_series: TimeSeries object
        """
        super().__init__()
        self.time_series = time_series
        self.prior_mean = None
        #use the default prior for the std
        self.prior_cov_chol = (target.get_parameter_std_prior()
            * np.ones(self.get_n_dim()))
        self.cp_parameter_before = None

        parameter_name_array = self.time_series.get_parameter_vector_name()
        self.prior_mean = target.get_parameter_mean_prior(parameter_name_array)

    #implemented
    def get_n_dim(self):
        return self.time_series.n_parameter

    #implemented
    def get_state(self):
        return self.time_series.get_parameter_vector()

    #implemented
    def update_state(self, state):
        self.time_series.set_parameter_vector(state)
        self.time_series.update_all_cp_parameters()

    #implemented
    def get_log_likelihood(self):
        return self.time_series.get_joint_log_likelihood()

    #implemented
    def get_log_target(self):
        ln_l = self.get_log_likelihood()
        ln_prior = self.get_log_prior()
        return ln_l + ln_prior

    def get_log_prior(self):
        return np.sum(
            stats.norm.logpdf(
                self.get_state(), self.prior_mean, self.prior_cov_chol))

    #implemented
    def save_state(self):
        self.cp_parameter_before = self.time_series.copy_parameter()

    #implemented
    def revert_state(self):
        self.time_series.set_parameter(self.cp_parameter_before)

    #implemented
    def simulate_from_prior(self, rng):
        return rng.normal(self.prior_mean, self.prior_cov_chol)

    #implemented
    def get_prior_mean(self):
        return self.prior_mean

class TargetZ(target.Target):
    """Contains the posterior distribution for the latent variables z

    Attributes:
        time_series: TimeSeries object being wrapped around
        z_array_before: copy of time_series.z_array when save_state called
    """

    def __init__(self, time_series):
        super().__init__()
        self.time_series = time_series
        self.z_array_before = None

    #implemented
    def get_n_dim(self):
        return len(self.time_series)

    #implemented
    def get_state(self):
        return self.time_series.z_array

    #implemented
    def update_state(self, state):
        self.time_series.z_array = state
        self.time_series.update_all_cp_parameters()

    #implemented
    def get_log_likelihood(self):
        return self.time_series.get_joint_log_likelihood()

    #implemented
    def get_log_target(self):
        return self.get_log_likelihood()

    #implemented
    def save_state(self):
        self.z_array_before = self.time_series.z_array.copy()

    #implemented
    def revert_state(self):
        self.update_state(self.z_array_before)

class TargetPrecision(target.Target):
    """Contains the posterior of the parameter precision

    This class can assess an instance of TargetParameter via self.time_series.
        See update_state() and prograte_precision().

    Attributes:
        time_series: TimeSeries object
        prior: dictionary containing 2 distributions, random_state are accessed,
            rvs() are logpdf() are called. E.g. use distributions from
            scipy.stats. 0th prior for regression parameter, 1st for ARMA
            parameter. The keys are "precision_reg" and "precision_arma" or see
            target.get_precision_prior().
        precision: 2 vector, state of the chain
        precision_before: copy of precision
        arma_index: array of boolean, pointing to each parameter, True if it the
            parameter is an ARMA term
    """

    def __init__(self, time_series):
        super().__init__()
        self.time_series = time_series
        self.prior = target.get_precision_prior()
        self.precision = []
        self.precision_before = None
        self.arma_index = None

        #initalise precision using the prior mean
        for prior in self.prior.values():
            self.precision.append(prior.mean())
        self.precision = np.asarray(self.precision)

        #initalise arma_index
        self.arma_index = target.get_arma_index(
            self.time_series.get_parameter_vector_name())

    def get_cov_chol(self):
        """Return the vector of parameter prior std

        Returns:
            vector, element of each parameter, prior std
        """
        cov_chol = []
        for is_arma in self.arma_index:
            if is_arma:
                cov_chol.append(1 / self.precision[1])
            else:
                cov_chol.append(1 / self.precision[0])
        cov_chol = np.sqrt(np.asarray(cov_chol))
        return cov_chol

    #implemented
    def get_n_dim(self):
        return len(self.precision)

    #implemented
    def get_state(self):
        return self.precision

    #implemented
    def update_state(self, state):
        self.precision = state.copy()
        self.prograte_precision()

    #implemented
    def get_log_likelihood(self):
        return self.time_series.parameter_target.get_log_prior()

    #implemented
    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        ln_prior = 0
        for i, prior in enumerate(self.prior.values()):
            ln_prior += prior.logpdf(self.precision[i])
        return ln_prior

    #implemented
    def save_state(self):
        self.precision_before = self.precision.copy()

    #implemented
    def revert_state(self):
        self.precision = self.precision_before
        self.prograte_precision()

    #implemented
    def simulate_from_prior(self, rng):
        prior_simulate = []
        for prior in self.prior.values():
            prior.random_state = rng
            prior_simulate.append(prior.rvs())
        return np.asarray(prior_simulate)

    def prograte_precision(self):
        """Update the std vector in time_series.parameter_target with the
            current state
        """
        self.time_series.parameter_target.prior_cov_chol = self.get_cov_chol()
