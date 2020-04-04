import numpy as np
from scipy import stats

from compound_poisson.mcmc import target

class TargetParameter(target.Target):
    """Wrapper Target class to evaluate the posterior of the parameter

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
        self.prior_cov_chol = (target.get_parameter_std_prior()
            * np.ones(self.get_n_dim()))
        self.cp_parameter_before = None

        parameter_name_array = self.time_series.get_parameter_vector_name()
        self.prior_mean = target.get_parameter_mean_prior(parameter_name_array)

    def get_n_dim(self):
        return self.time_series.n_parameter

    def get_state(self):
        return self.time_series.get_parameter_vector()

    def update_state(self, state):
        self.time_series.set_parameter_vector(state)
        self.time_series.update_all_cp_parameters()

    def get_log_likelihood(self):
        return self.time_series.get_joint_log_likelihood()

    def get_log_target(self):
        ln_l = self.get_log_likelihood()
        ln_prior = self.get_log_prior()
        return ln_l + ln_prior

    def get_log_prior(self):
        return np.sum(
            stats.norm.logpdf(
                self.get_state(), self.prior_mean, self.prior_cov_chol))

    def save_state(self):
        self.cp_parameter_before = self.time_series.copy_parameter()

    def revert_state(self):
        self.time_series.set_parameter(self.cp_parameter_before)

    def simulate_from_prior(self, rng):
        return rng.normal(self.prior_mean, self.prior_cov_chol)

    def get_prior_mean(self):
        return self.prior_mean

class TargetZ(target.Target):
    """Wrapper Target class for the latent variables z

    Attributes:
        time_series: TimeSeries object being wrapped around
        z_array_before: copy of time_series.z_array when save_state called
    """

    def __init__(self, time_series):
        super().__init__()
        self.time_series = time_series
        self.z_array_before = None

    def get_n_dim(self):
        return len(self.time_series)

    def get_state(self):
        return self.time_series.z_array

    def update_state(self, state):
        self.time_series.z_array = state
        self.time_series.update_all_cp_parameters()

    def get_log_likelihood(self):
        return self.time_series.get_joint_log_likelihood()

    def get_log_target(self):
        return self.get_log_likelihood()

    def save_state(self):
        self.z_array_before = self.time_series.z_array.copy()

    def revert_state(self):
        self.update_state(self.z_array_before)

class TargetPrecision(target.Target):
    """Wrapper Target class to evaluate the posterior of the precision of the
        parameter prior covariance

    Attributes:
        parameter_target: TargetParameter object
        prior: array containing 2 distributions, random_state are accessed,
            rvs() are logpdf() are called. E.g. use distributions from
            scipy.stats. 0th prior for regression parameter, 1st for ARMA
            parameter
        precision: 2 vector
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

    def get_n_dim(self):
        return len(self.precision)

    def get_state(self):
        return self.precision

    def update_state(self, state):
        self.precision = state.copy()
        self.prograte_precision()

    def get_log_likelihood(self):
        return self.time_series.parameter_target.get_log_prior()

    def get_log_target(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        ln_prior = 0
        for i, prior in enumerate(self.prior.values()):
            ln_prior += prior.logpdf(self.precision[i])
        return ln_prior

    def save_state(self):
        self.precision_before = self.precision.copy()

    def revert_state(self):
        self.precision = self.precision_before
        self.prograte_precision()

    def simulate_from_prior(self, rng):
        prior_simulate = []
        for prior in self.prior:
            prior.random_state = rng
            prior_simulate.append(prior.rvs())
        return np.asarray(prior_simulate)

    def prograte_precision(self):
        self.time_series.parameter_target.prior_cov_chol = self.get_cov_chol()
