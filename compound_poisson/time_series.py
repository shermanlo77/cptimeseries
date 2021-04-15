import math
from os import path

import cycler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pandas
import pandas.plotting
from statsmodels.tsa import stattools
from scipy import special

from compound_poisson import arma
from compound_poisson import forecast
from compound_poisson import parameter
from compound_poisson import terms


class TimeSeries(object):
    """Compound Poisson Time Series with ARMA behaviour

    A time series distribued as compound Poisson with dynamic varying
        parameters. Objects from this class can simulate the model and/or fit
        onto provided data or simulated data
    Initalise the model by passing model fields, rainfall data and the number
        of ARMA terms for the Poisson rate and the gamma mean.
    Alternatively, initalise the model by passing model fields and an array of
        Parameters (see the __init__)
    __len__(), __iter__(), __getitem__() and __setitem__() overridden to
        correspond to the array of rainfall
    Subclasses should implement the method fit() to fit the model onto the data

    Attributes:
        x: np.array as a design matrix of the model fields, shape
            (n, n_model_field)
        x_shift: mean of x along time
        x_scale: standard deviation of x along time
        model_field_name: array containing a name (string) for each model field
        time_array: time stamp for each time step (default range(n))
        n_model_field: number of dimensions of the model fields
        poisson_rate: PoissonRate object containing the parameter at each time
            step and parameters for the regressive and ARMA parameters elements
        gamma_mean:  GammaMean containing the parameter at each time step and
            parameters for the regressive and ARMA parameters elements
        gamma_dispersion:  GammaDispersion containing the parameter at each
            time step and parameters for the regressive and ARMA parameters
            elements
        n_parameter: number of total (from poisson_rate, gamma_mean,
            gamma_dispersion) reg parameters
        cp_parameter_array: array pointing to poisson_rate, gamma_mean and
            gamma_dispersion
        z_array: latent poisson variables at each time step
        z_var_array: variance of the z at each time step
        y_array: compound poisson variables at each time step
        fitted_time_series: time series before this one, used for forecasting
        rng: numpy.random.RandomState object
        id: default None, 2 elemenet array containing latitude and longitiude
            index, used by Downscale
        forecaster: forecast.time_series.Forecaster object, contains forecasts
            on a test set
        self_forecaster: forecast.time_series.SelfForecaster object, contains
            forecasts on the training set
        forecaster_memmap_dir: directory to store forecast objects
        forecaster_rng: rng for method instantiate_forecast_self()
        self_forecaster_rng: rng for method instantiate_forecast()
    """

    def __init__(self,
                 x,
                 rainfall=None,
                 poisson_rate_n_arma=None,
                 gamma_mean_n_arma=None,
                 cp_parameter_array=None):
        """
        Provide the following combination parameters (or signature) only:
            -x, rainfall, poisson_rate_n_arma, gamma_mean_n_arma
                -to be used for fitting the model onto a provided data
            -x, rainfall, cp_parameter_array
                -to be used for fitting the model onto a provided data with a
                    provided initial value
            -x, cp_parameter_array
                -to be used for simulating rainfall
            -x, poisson_rate_n_arma, gamma_mean_n_arma
                -to be used for simulating rainfall using the default
                parameters

        Args:
            x: design matrix of the model fields, shape (n, n_model_field)
            rainfall: array of rainfall data. If none, all rain is zero
            poisson_rate_n_arma: 2 element array, number of AR and MA terms for
                the poisson rate. Ignored if cp_parameter_array is provided.
            gamma_mean_n_arma: 2 element array, number of AR and MA terms for
                the gamma mean. Ignored if cp_parameter_array is provided.
            cp_parameter_array: array containing in order PoissonRate object,
                GammaMean object, GammaDispersion object
        """

        if type(x) is pandas.core.frame.DataFrame:
            self.x = np.asarray(x)
        else:
            self.x = x
        self.x_shift = np.mean(self.x, 0)
        self.x_scale = np.std(self.x, 0, ddof=1)
        n = self.x.shape[0]
        self.model_field_name = []

        self.time_array = range(n)
        self.n_model_field = self.x.shape[1]
        self.poisson_rate = None
        self.gamma_mean = None
        self.gamma_dispersion = None
        self.n_parameter = None
        # array containing poisson_rate, gamma_mean and gamma_dispersion
        self.cp_parameter_array = None
        self.z_array = np.zeros(n)  # z_array can be float in E step
        self.z_var_array = np.zeros(n)
        self.y_array = None
        self.fitted_time_series = None
        self.rng = None
        self.id = None
        self.forecaster = None
        self.self_forecaster = None
        self.forecaster_memmap_dir = ""
        self.forecaster_rng = None
        self.self_forecaster_rng = None

        self.set_rng(random.SeedSequence())

        # name the model fields, or extract from pandas data frame
        if type(x) is pandas.core.frame.DataFrame:
            self.model_field_name = x.columns
        else:
            for i in range(self.n_model_field):
                self.model_field_name.append("model_field_" + str(i))

        if rainfall is None:
            self.y_array = np.zeros(n)
        else:
            self.y_array = rainfall

        # initalise parameters if none is provided, all regression parameters
        # to zero, constant is a naive estimate
        if cp_parameter_array is None:
            # cannot estimate parameter if no rain
            if rainfall is None:
                poisson_rate = parameter.PoissonRate(
                    self.n_model_field, poisson_rate_n_arma)
                gamma_mean = parameter.GammaMean(
                    self.n_model_field, gamma_mean_n_arma)
                gamma_dispersion = parameter.GammaDispersion(
                    self.n_model_field)
                cp_parameter_array = [
                    poisson_rate,
                    gamma_mean,
                    gamma_dispersion,
                ]
                self.set_new_parameter(cp_parameter_array)
            else:
                self.initalise_parameters(
                    poisson_rate_n_arma, gamma_mean_n_arma)
        else:
            self.set_new_parameter(cp_parameter_array)

    def initalise_parameters(self, poisson_rate_n_arma, gamma_mean_n_arma):
        """Set the initial parameters

        Modifies poisson_rate, gamma_mean, gamma_dispersion and
            cp_parameter_array
        Set the initial parameters to be naive estimates using method of
            moments and y=0 count

        Args:
            poisson_rate_n_arma: 2 element array (n_ar, n_ma) for the Poisson
                rate
            gamma_mean_n_arma: 2 element array (n_ar, n_na) for the Gamma mean
        """
        y_array = self.y_array
        n = len(self)
        n_model_field = self.n_model_field
        # estimate the parameters assuming the data is iid, use method of
        # moments estimators
        poisson_rate_guess = math.log(
            (n + 0.5) / (n - np.count_nonzero(y_array) + 1))
        gamma_mean_guess = np.mean(y_array) / poisson_rate_guess
        gamma_dispersion_guess = (np.var(y_array, ddof=1)
                                  / poisson_rate_guess
                                  / math.pow(gamma_mean_guess, 2)
                                  - 1)
        if gamma_dispersion_guess <= 0:
            print("Warning! gamma_dispersion_guess is negative with value",
                  gamma_dispersion_guess)
            gamma_dispersion_guess = abs(gamma_dispersion_guess)
        # instantise parameters and set it
        poisson_rate = parameter.PoissonRate(
            n_model_field, poisson_rate_n_arma)
        gamma_mean = parameter.GammaMean(n_model_field, gamma_mean_n_arma)
        gamma_dispersion = parameter.GammaDispersion(n_model_field)
        poisson_rate["const"] = math.log(poisson_rate_guess)
        gamma_mean["const"] = math.log(gamma_mean_guess)
        gamma_dispersion["const"] = math.log(gamma_dispersion_guess)
        self.set_new_parameter([poisson_rate, gamma_mean, gamma_dispersion])

    def initalise_parameters_given_arma(self):
        """Set the initial parameters (using current number ARMA parameters)

        Modifies poisson_rate, gamma_mean, gamma_dispersion and
            cp_parameter_array
        Set the initial parameters to be naive estimates using method of
            moments and y=0 count with the current number of ARMA parameters
        """
        poisson_rate_n_arma = (self.poisson_rate.n_ar, self.poisson_rate.n_ma)
        gamma_mean_n_arma = (self.gamma_mean.n_ar, self.gamma_mean.n_ma)
        self.initalise_parameters(poisson_rate_n_arma, gamma_mean_n_arma)

    def get_normalise_x(self, index):
        """Return a model field vector at a specific time step which is
            normalised to have mean zero, std one along the time axis
        """
        return (self.x[index] - self.x_shift) / self.x_scale

    def set_new_parameter(self, cp_parameter_array):
        """Set the member variables of newly instantised cp paramters

        Set the member variables of poisson_rate, gamma_mean,
            gamma_dispersion and cp_parameter_array. Itself is assigned as the
            parent to these objects. The reg parameters in these objects are
            converted to numpy array
        """
        self.set_parameter(cp_parameter_array)
        for parameter_i in self.cp_parameter_array:
            # assign the parameters parents as self, this is so that the
            # parameter objects has access to all member variables and
            # methods
            parameter_i.assign_parent(self)
            # prepare the parameters by converting all to numpy array
            parameter_i.convert_all_to_np()
        # get number of parameters in this model
        self.n_parameter = 0
        for parameter_i in self.cp_parameter_array:
            for reg in parameter_i.values():
                self.n_parameter += reg.shape[0]

    def set_parameter(self, cp_parameter_array):
        """Set the member variables of the cp paramters

        Set the member variables of poisson_rate, gamma_mean, gamma_dispersion
            and cp_parameter_array.

        Args:
            array containing in order: PoissonRate object, GammaMean object,
                GammaDispersion object
        """
        self.poisson_rate = cp_parameter_array[0]
        self.gamma_mean = cp_parameter_array[1]
        self.gamma_dispersion = cp_parameter_array[2]
        self.cp_parameter_array = cp_parameter_array

    def copy_parameter(self):
        """Deep copy compound Poisson parameters

        Copy the compound Poisson parameters poisson_mean, gamma_mean and
            gamma_dispersion. Deep copies the regression parameters and also
            the value of itself at each time step

        Returns:
            array containing in order: poisson_mean, gamma_mean,
                gamma_dispersion
        """
        cp_parameter_array = []
        for parameter_i in self.cp_parameter_array:
            cp_parameter_array.append(parameter_i.copy())
        return cp_parameter_array

    def copy_parameter_only_reg(self):
        """Deep copy the regression parameters of compound Poisson parameters

        Copy the compound Poisson parameters poisson_mean, gamma_mean and
            gamma_dispersion. Deep copies the regression parameters only

        Returns:
            array containing in order: poisson_mean, gamma_mean,
                gamma_dispersion
        """
        cp_parameter_array = []
        for parameter_i in self.cp_parameter_array:
            cp_parameter_array.append(parameter_i.copy_reg())
        return cp_parameter_array

    def get_parameter_vector(self):
        """Return the regression parameters as one vector
        """
        # for each parameter, concatenate vectors together
        parameter_vector = np.zeros(0)
        for parameter_i in self.cp_parameter_array:
            parameter_vector = np.concatenate(
                (parameter_vector, parameter_i.get_reg_vector()))
        return parameter_vector

    def get_parameter_vector_name(self):
        """Return the name of each element of the regression parameter vector
        """
        vector_name = []
        for parameter_i in self.cp_parameter_array:
            parameter_name = parameter_i.get_reg_vector_name()
            for name in parameter_name:
                if self.id is None:
                    vector_name.append(name)
                else:
                    vector_name.append(str(self.id) + "_" + name)
        return vector_name

    def set_parameter_vector(self, parameter_vector):
        """Set the regression parameters using a single vector
        """
        parameter_counter = 0  # count the number of parameters
        # for each parameter, take the correction section of parameter_vector
        # and put it in the parameter
        for parameter_i in self.cp_parameter_array:
            n_parameter = 0
            for reg in parameter_i.values():
                n_parameter += reg.shape[0]
            parameter_i.set_reg_vector(
                parameter_vector[
                    parameter_counter:parameter_counter+n_parameter])
            parameter_counter += n_parameter

    def simulate_i(self, i):
        """Simulate a time step of the time series

        MODIFIES ITSELF
        Because of ARMA behaviour, this method should be called sequentially.
            Modify the member variables poisson_rate, gamma_mean,
            gamma_dispersion, z_array and y_array.
        """
        self.update_cp_parameters(i)
        # simulate this compound Poisson
        self[i], self.z_array[i] = self.simulate_cp(
            self.poisson_rate[i], self.gamma_mean[i],
            self.gamma_dispersion[i])

    def simulate(self):
        """Simulate a whole time series

        MODIFIES ITSELF
        Simulate a time series given the model fields self.x and cp parameters.
            Modify the member variables poisson_rate, gamma_mean and
            gamma_dispersion by updating its values at each time step.
            Also modifies self.z_array and self.y_array with the
            simulated values.
        """
        # simulate n times
        for i in range(len(self)):
            self.simulate_i(i)

    def simulate_given_z(self):
        """Return a simulated of a whole time series with given z

        MODIFIES ITSELF
        Simulate a time series given the model fields self.x, cp parameters and
            latent variables in self.z_array. Modify the member variables
            poisson_rate, gamma_mean and gamma_dispersion by updating its
            values at each time step. Also modifies self.y_array with the
            simulated values.
        """
        for i in range(len(self)):
            # get the parameters of the compound Poisson at this time step
            self.update_cp_parameters(i)
            # simulate this compound Poisson
            self[i] = self.simulate_cp_given_z(
                self.z_array[i], self.gamma_mean[i], self.gamma_dispersion[i])

    def simulate_cp(self, poisson_rate, gamma_mean, gamma_dispersion):
        """Simulate a single compound poisson random variable

        Args:
            poisson_rate: scalar
            gamma_mean: scalar
            gamma_dispersion: scalar

        Returns:
            tuple contain vectors of y (compound Poisson random variable) and z
                (latent Poisson random variable)
        """
        # poisson random variable variable
        z = self.rng.poisson(poisson_rate)
        # gamma random variable
        y = self.simulate_cp_given_z(z, gamma_mean, gamma_dispersion)
        return (y, z)

    def simulate_cp_given_z(self, z, gamma_mean, gamma_dispersion):
        """Simulate a single compound poisson random varible given z

        Args:
            z: latent poisson variable
            gamma_mean: mean of the gamma random variable
            gamma_dispersion: dispersion of the gamma random variable

        Returns:
            simulated compound Poisson random variable
        """
        shape = z / gamma_dispersion  # gamma shape parameter
        scale = gamma_mean * gamma_dispersion  # gamma scale parameter
        y = self.rng.gamma(shape, scale=scale)  # gamma random variable
        return y

    def fit(self):
        """Fit the model onto the data

        To be implemented by subclasses
        """
        pass

    def update_all_cp_parameters(self):
        """Update all the member variables of the cp variables for all time
        steps

        See the method update_cp_parameters(self, index)
        """
        for i in range(len(self)):
            self.update_cp_parameters(i)

    def update_cp_parameters(self, index):
        """Update the variables of the cp variables

        Updates and modifies the Poisson rate, gamma mean and gama dispersion
            parameters at a given time step

        Args:
            index: the time step to update the parameters at
        """
        for parameter_i in self.cp_parameter_array:
            parameter_i.update_value_array(index)

    def get_joint_log_likelihood(self):
        """Joint log likelihood

        Returns the joint log likelihood of the compound Poisson time series
            assuming the latent variable z are observed (via simulation or
            estimating using the E step). Requires the method
            update_all_cp_parameters() to be called beforehand or
            update_cp_parameters(index) for index in range(len(self)) if only a
            few parameters has changed. Note that this is done after calling
            e_step(), thus this method can be called without any prerequisites
            afer calling e_step().

        Returns:
            log likelihood
        """
        ln_l_array = np.zeros(len(self))
        for i in range(len(self)):
            ln_l_array[i] = self.get_joint_log_likelihood_i(i)
        return np.sum(ln_l_array)

    def get_em_objective(self):
        """Return M step objective for a single data point

        Requires the method update_all_cp_parameters() to be called beforehand
            or update_cp_parameters(index) for index in range(len(self)).
        """
        ln_l_array = np.zeros(len(self))
        for i in range(len(self)):
            ln_l_array[i] = self.get_em_objective_i(i)
        return np.sum(ln_l_array)

    def get_em_objective_i(self, i):
        """Return M step objective for a single data point

        Requires the method update_all_cp_parameters() to be called beforehand
            or update_cp_parameters(index) for index in range(len(self)).
        """
        objective = self.get_joint_log_likelihood_i(i)
        z = self.z_array[i]
        if z > 0:
            z_var = self.z_var_array[i]
            gamma_dispersion = self.gamma_dispersion[i]
            objective -= (0.5*z_var*special.polygamma(1, z/gamma_dispersion)
                          / math.pow(gamma_dispersion, 2))
        return objective

    def get_joint_log_likelihood_i(self, i):
        """Return joint log likelihood for a single data point

        Requires the method update_all_cp_parameters() to be called beforehand
            or update_cp_parameters(index) for index in range(len(self)).
        """
        ln_l = -self.poisson_rate[i]
        z = self.z_array[i]
        if z > 0:
            y = self[i]
            gamma_mean = self.gamma_mean[i]
            gamma_dispersion = self.gamma_dispersion[i]
            cp_term = terms.Terms(self, i)
            term_array = np.zeros(3)
            term_array[0] = -y/gamma_mean/gamma_dispersion
            term_array[1] = -math.log(y)
            term_array[2] = cp_term.ln_wz(z)
            ln_l += np.sum(term_array)
        return ln_l

    def e_step(self):
        """Does the E step of the EM algorithm

        Make estimates of the z and updates the poisson rate, gamma mean and
            gamma dispersion parameters for each time step. Modifies the member
            variables z_array, poisson_rate, gamma_mean, and gamma_dispersion
        """
        # for each data point (forwards in time)
        for i in range(len(self)):
            # update the parameter at this time step
            self.update_cp_parameters(i)
            # if the rainfall is zero, then z is zero (it has not rained)
            if self[i] == 0:
                self.z_array[i] = 0
                self.z_var_array[i] = 0
            else:
                # work out the normalisation constant for the expectation
                sum = terms.Terms(self, i)
                normalisation_constant = sum.ln_sum_w()
                # work out the expectation
                sum = terms.TermsZ(self, i)
                self.z_array[i] = math.exp(
                    sum.ln_sum_w() - normalisation_constant)
                sum = terms.TermsZ2(self, i)
                self.z_var_array[i] = (math.exp(
                    sum.ln_sum_w() - normalisation_constant)
                    - math.pow(self.z_array[i], 2))

    def forecast_self(self, n_simulation):
        """Forecast itself

        Forecast itself given its parameters and estimates of z. Use
            expectation of the predictive distribution using Monte Carlo.
            Results are stored in self_forecaster

        Args:
            n_simulation: number of Monte Carlo simulations
        """
        if self.self_forecaster is None:
            self.self_forecaster = forecast.time_series.SelfForecaster(
                self, self.forecaster_memmap_dir)
            self.self_forecaster.start_forecast(n_simulation)
        else:
            self.self_forecaster.resume_forecast(n_simulation)

    def instantiate_forecast_self(self):
        """Instantiate TimeSeries for forecasting itself

        Instantiate TimeSeries object which is a copy of itself. Deep copy the
            parameters and z_array. Soft copy x.

        Returns:
            forecast: Forecast object
        """
        forecast = TimeSeries(
            self.x, cp_parameter_array=self.copy_parameter_only_reg())
        forecast.z_array = self.z_array.copy()
        forecast.x_shift = self.x_shift
        forecast.x_scale = self.x_scale
        forecast.model_field_name = self.model_field_name
        forecast.time_array = self.time_array
        forecast.rng = self.self_forecaster_rng
        return forecast

    def forecast(self, x, n_simulation):
        """Forecast

        Forecast itself given its parameters, estimates of z and future model
            fields. Use expectation of the predictive distribution using Monte
            Carlo. Results are stored in forecaster

        Args:
            x: model fields, np.array, each element for each time step
            n_simulation: number of Monte Carlo simulations
        """
        if self.forecaster is None:
            self.forecaster = forecast.time_series.Forecaster(
                self, self.forecaster_memmap_dir)
            self.forecaster.start_forecast(n_simulation, x)
        else:
            self.forecaster.resume_forecast(n_simulation)

    def instantiate_forecast(self, x):
        """Instantiate TimeSeries for simulating the future given the current
            parameters

        Instantiate TimeSeries object containing future model fields. Deep copy
            the parameters. This method is used by simulate_future() and
            instantiate_forecast()

        Returns:
            forecast: TimeSeries object
        """
        forecast = TimeSeries(
            x, cp_parameter_array=self.copy_parameter_only_reg())
        forecast.cast_arma(arma.ArmaForecast)
        forecast.x_shift = self.x_shift
        forecast.x_scale = self.x_scale
        forecast.fitted_time_series = self
        forecast.model_field_name = self.model_field_name
        forecast.time_array = []
        forecast.rng = self.forecaster_rng
        # the forecast time_array is the future
        last_time = self.time_array[len(self)-1]
        time_diff = last_time - self.time_array[len(self)-2]
        for i in range(len(x)):
            forecast.time_array.append(last_time + (i+1)*time_diff)

        return forecast

    def set_rng(self, seed_sequence):
        """Set rng
        """
        rng_array = []
        for s in seed_sequence.spawn(3):
            rng_s = random.RandomState(random.MT19937(s))
            rng_array.append(rng_s)
        self.rng = rng_array[0]
        self.forecaster_rng = rng_array[1]
        self.self_forecaster_rng = rng_array[2]

    def cast_arma(self, arma_class):
        """Cast the arma object

        Update the member variable arma to be of another type using a provided
            class

        Args:
            arma_class: class object, self will be passed into the constructor
        """
        for parameter_i in self.cp_parameter_array:
            parameter_i.cast_arma(arma_class)

    def print_figures(self, directory, prefix=""):
        """Print figures

        Save figure for precipitation time series
        Save figure for cumulative frequency of precipitation in the time
            series
        Save figures for sample autocorrelation and partial autocorrelation
        Save figures for the poisson rate, gamma mean, gamma dispersion, latent
            variable Z over time
        Save the TimeSeries in text (converted to string which shows
            parameters)

        Args:
            directory: where to save the figures
            prefix: what to name the figures
        """

        # required when plotting times on an axis
        pandas.plotting.register_matplotlib_converters()

        colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        cycle = cycler.cycler(color=[colours[0]], linewidth=[1])

        # get autocorrelations
        acf = stattools.acf(self.y_array, nlags=20, fft=True)
        try:
            pacf = stattools.pacf(self.y_array, nlags=20)
        except(stattools.LinAlgError):
            pacf = np.full(21, np.nan)

        # print precipitation time series
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle)
        plt.plot(self.time_array, self.y_array)
        plt.xlabel("time")
        plt.ylabel("rainfall (mm)")
        plt.savefig(path.join(directory, prefix + "rainfall.pdf"))
        plt.close()

        # print precipitation cumulative frequency
        # draw dot for mass at 0 mm
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle)
        rain_sorted = np.sort(self.y_array)
        cdf = np.asarray(range(len(self)))
        plt.plot(rain_sorted, cdf)
        if np.any(rain_sorted == 0):
            non_zero_index = rain_sorted.nonzero()[0]
            if non_zero_index.size > 0:
                non_zero_index = rain_sorted.nonzero()[0][0] - 1
            else:
                non_zero_index = len(cdf) - 1
            plt.scatter(0, cdf[non_zero_index])
        plt.xlabel("rainfall (mm)")
        plt.ylabel("cumulative frequency")
        plt.savefig(path.join(directory, prefix + "cdf.pdf"))
        plt.close()

        # plot sample autocorrelation
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle)
        plt.bar(np.asarray(range(acf.size)), acf)
        plt.axhline(1/math.sqrt(len(self)), linestyle='--', linewidth=1)
        plt.axhline(-1/math.sqrt(len(self)), linestyle='--', linewidth=1)
        plt.xlabel("time (day)")
        plt.ylabel("autocorrelation")
        plt.savefig(path.join(directory, prefix + "acf.pdf"))
        plt.close()

        # plot sample partial autocorrelation
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle)
        plt.bar(np.asarray(range(pacf.size)), pacf)
        plt.axhline(1/math.sqrt(len(self)), linestyle='--', linewidth=1)
        plt.axhline(-1/math.sqrt(len(self)), linestyle='--', linewidth=1)
        plt.xlabel("time (day)")
        plt.ylabel("partial autocorrelation")
        plt.savefig(path.join(directory, prefix + "pacf.pdf"))
        plt.close()

        # plot the poisson rate over time
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle)
        plt.plot(self.time_array, self.poisson_rate.value_array)
        plt.xlabel("time")
        plt.ylabel("poisson rate")
        plt.savefig(path.join(directory, prefix + "poisson_rate.pdf"))
        plt.close()

        # plot the gamma mean over time
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle)
        plt.plot(self.time_array, self.gamma_mean.value_array)
        plt.xlabel("time")
        plt.ylabel("gamma mean (mm)")
        plt.savefig(path.join(directory, prefix + "gamma_mean.pdf"))
        plt.close()

        # plot the gamme dispersion over time
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle)
        plt.plot(self.time_array, self.gamma_dispersion.value_array)
        plt.xlabel("time")
        plt.ylabel("gamma dispersion")
        plt.savefig(path.join(directory, prefix + "gamma_dispersion.pdf"))
        plt.close()

        # plot the latent variable z over time
        plt.figure()
        ax = plt.gca()
        ax.set_prop_cycle(cycle)
        plt.plot(self.time_array, self.z_array)
        plt.xlabel("time")
        plt.ylabel("Z")
        plt.savefig(path.join(directory, prefix + "z.pdf"))
        plt.close()

        # print the parameters in text
        file = open(path.join(directory, prefix + "parameter.txt"), "w")
        file.write(str(self))
        file.close()

    def __str__(self):
        # return the reg parameters for each cp parameter
        string = ""
        for i in range(len(self.cp_parameter_array)):
            parameter_i = self.cp_parameter_array[i]
            string += parameter_i.__class__.__name__
            string += ":"
            string += parameter_i.__str__()
            if i != len(self.cp_parameter_array) - 1:
                string += "\n"
        return string

    def __iter__(self):
        return self.y_array.__iter__()

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, index):
        return self.y_array[index]

    def __setitem__(self, index, value):
        self.y_array[index] = value


def static_update_all_cp_parameters(time_series):
    """Static version of update_all_cp_parameters(). Used for parallel
        programming.
    """
    time_series.update_all_cp_parameters()
    return time_series
