"""Classes for time varying parameters in the compound Poisson model

Classes for time varying parameters in the compound Poisson model, eg Poisson
    rate, Gamma mean and Gamma dispersion. The parameters for the ARMA
    emulation are defined here. The value of the parameter at each stage are
    stored.

Parameter
    <- PoissonRate
    <- GammaMean
    <- GammaDispersion

TimeSeries
    <>-1 PoissonRate
    <>-1 GammaMean
    <>-1 GammaDispersion
"""

import math

import numpy as np
from scipy import special

from compound_poisson import arma


class Parameter(object):
    """Dynamic parameters of the compound Poisson

    An abstract class
    Contains the parameter at each time step as it envolves. Instances from
        this class also contains regressive parameters, such as regressive,
        autoregressive, moving average and constant for the compound
        Poisson time series model
    Values of itself from the time series can be obtained or set using self[i]
        where i is an integer representing the time step
    Reg parameters can be obtained or set using self[key] where key can be
        "reg", "AR", "MA" or "const"
    Methods keys(), values(), items() return the same method from
        self.reg_parameters, that is, return iterators of the reg parameters
    Has length corresponding to the length of the time series

    This is an abstract class with the following methods need to be
        implemented:
        -ma_term(self, index)
        -d_self_ln_l(self)
        -d_reg_ma(self, index, key)

    Attributes:
        n_model_field: number of dimensions the model fields has
        n_ar: number of autoregressive terms
        n_ma: number of moving average terms
        reg_parameters: dictionary containing regressive (reg) parameters
            with keys "reg", "AR", "MA", "const" which corresponds to the
            names of the reg parameters
        arma: object to evalute arma terms
        value_array: array of values of the parameter for each time step
        time_series: TimeSeries object containing this
        _d_reg_self_array: dictionary containing arrays of derivates
            of itself wrt reg parameters
    """

    def __init__(self, n_model_field, n_arma):
        """
        Args:
            n_model_field: number of model fields
            n_arma: 2 element array, (n_ar, n_ma)
        """
        self.n_model_field = n_model_field
        if n_arma is None:
            self.n_ar = 0
            self.n_ma = 0
        else:
            self.n_ar = n_arma[0]
            self.n_ma = n_arma[1]
        self.reg_parameters = {
            "reg": np.zeros(n_model_field),
            "const": 0.0,
        }
        self.arma = None
        self.value_array = None
        self.time_series = None
        self._d_reg_self_array = None
        # add AR and MA parameters if there are at least one
        if self.n_ar > 0:
            self.reg_parameters["AR"] = np.zeros(self.n_ar)
        if self.n_ma > 0:
            self.reg_parameters["MA"] = np.zeros(self.n_ma)

    def assign_parent(self, parent):
        """Assign parent

        Assign the member variable time_series which points to the TimeSeries
            object which owns self
        """
        self.time_series = parent
        self.value_array = np.zeros(len(parent))
        self.arma = arma.Arma(self)

    def copy(self):
        """Return deep copy of itself
        """
        copy = self.copy_reg()
        copy.assign_parent(self.time_series)
        copy.value_array = self.value_array.copy()
        return copy

    def copy_reg(self):
        """Return copy of itself containing a deep copy of the regression
            parameters
        """
        # GammaDispersion does not hava ARMA terms
        if isinstance(self, GammaDispersion):
            copy = self.__class__(self.n_model_field)
        # any other parameters require ARMA terms
        else:
            copy = self.__class__(self.n_model_field, (self.n_ar, self.n_ma))
        # deep copy regression parameters
        for key, value in self.reg_parameters.items():
            if type(value) is np.ndarray:
                copy[key] = value.copy()
            else:
                copy[key] = value
        return copy

    def convert_all_to_np(self):
        """Convert all values in self.reg_parameters to be numpy array

        Required if the constant term is not a numpy array
        """
        for key, value in self.reg_parameters.items():
            self[key] = np.asarray(value)
            if key == "reg":
                self[key] = np.reshape(self[key], self.n_model_field)
            elif key == "AR":
                self[key] = np.reshape(self[key], self.n_ar)
            elif key == "MA":
                self[key] = np.reshape(self[key], self.n_ma)
            elif key == "const":
                self[key] = np.reshape(self[key], 1)

    def update_value_array(self, index):
        """Update the variables of value_array

        Updates and modifies the value_array at a given time step given all
            the z before that time step.

        Args:
            index: the time step to update the parameters at
        """

        # regressive on the model fields and constant
        exponent = 0.0
        exponent += self["const"]
        exponent += np.dot(self["reg"],
                           self.time_series.get_normalise_x(index))
        # add ARMA terms
        if "AR" in self.keys():
            exponent += np.dot(self["AR"], self.arma.ar_term(index))
        if "MA" in self.keys():
            exponent += np.dot(self["MA"], self.arma.ma_term(index))
        # exp link function, make it positive
        self[index] = math.exp(exponent)

    def cast_arma(self, arma_class):
        """Cast the arma object

        Update the member variable arma to be of another type using a provided
            subclass class

        Args:
            arma_class: class object, self will be passed into the constructor
        """
        self.arma = arma_class(self)

    def ar(self, index):
        """AR term for a specific time step

        Returns the autoregressive term for a specific time step, to be used by
            the arma object.
        """
        return math.log(self[index]) - self["const"]

    def ma(self, y, z, poisson_rate, gamma_mean, gamma_dispersion):
        """MA term given parameters

        Returns the moving average term given parameters, to be implemented. to
            be used by the arma object.
        """
        pass

    def gradient_descent(self, step_size):
        """Update itself using gradient descent

        Modifies self.reg_parameters using gradient_descent given the model
            fields and z_array. Requires calculate_d_reg_self_i() to be
            called for all index. See the method m_step in TimeSeriesGd

        Args:
            step_size: change parameters by step_size/len(self) * gradient
        """
        gradient = self.d_reg_ln_l()
        gradient *= step_size/len(self)
        self += gradient

    def stochastic_gradient_descent(self, index, step_size):
        """Update itself using stochastic gradient descent

        Modifies self.reg_parameters using gradient_descent given the model
            fields and z_array. Requires calculate_d_reg_self_i() to be
            called for all index up to the args index. See the method m_step
            in TimeSeriesSgd

        Args:
            index: the point in the time series to be used for stochastic
                gradient descent
            step_size: change parameters by step_size * gradient
        """
        d_self_ln_l = self.d_self_ln_l(index)
        for key in self.keys():
            self[key] += (step_size * (self._d_reg_self_array[key][index]
                          * d_self_ln_l))

    def d_reg_ln_l(self):
        """Returns the derivate of the log likelihood wrt reg parameters

        Returns a Parameter object with values corresponding to the derivate of
            the log likelihood wrt reg parameters. Requires the following
            methods to be called beforehand:
            -reset_d_reg_self_array()
            -calculate_d_reg_self_i(index) where the outer loop is
                for each time step and the inner loop is for each reg
                parameter. This is required because the derivate of, for
                example, GammaMean.ma[index] depends on
                GammaDispersion._d_reg_self_array[index-1]. As a result, the
                derivate of all reg parameters must be found together for
                each time step
            See the method TimeSeriesGd.m_step(self) to see encxample on how to
                call this method

        Returns:
            Parameter object with values corresponding to the derivate of the
                log likelihood wrt reg parameters with keys "reg", "const" as
                well as if available "AR" and "MA"
        """
        d_self_ln_l = self.d_self_ln_l_all()
        d_reg_ln_l = self._d_reg_self_array
        for key in self.keys():
            for i in range(len(self)):
                # chain rule
                d_reg_ln_l[key][i] = (
                    self._d_reg_self_array[key][i] * d_self_ln_l[i])
            # sum over all time steps (chain rule for partial diff)
            value = d_reg_ln_l[key]
            if value.ndim == 1:
                d_reg_ln_l[key] = np.sum(value)
            else:
                d_reg_ln_l[key] = np.sum(value, 0)
        # put the gradient in a Parameter object and return it
        gradient = Parameter(self.n_model_field, (self.n_ar, self.n_ma))
        gradient.reg_parameters = d_reg_ln_l
        return gradient

    def d_self_ln_l_all(self):
        """Derivate of the log likelihood wrt itself for all time steps

        See the method d_reg_ln_l() for any requirements

        Returns:
            vector of gradients
        """
        d_self_ln_l = np.zeros(len(self))
        for i in range(len(self)):
            d_self_ln_l[i] = self.d_self_ln_l(i)
        return d_self_ln_l

    def d_self_ln_l(self, index):
        """Derivate of the log likelihood wrt itself for a given time step

        Abstract method - subclasses to implement

        Args:
            index: time step

        Returns:
            gradient
        """
        pass

    def reset_d_reg_self_array(self):
        """Reset _d_reg_self_array

        Set all values in _d_reg_self_array to be numpy zeros
        """
        self._d_reg_self_array = {}
        for key, value in self.reg_parameters.items():
            self._d_reg_self_array[key] = np.zeros((len(self), value.size))

    def calculate_d_reg_self_i(self, index):
        """Calculates the derivate of itself wrt each reg parameter

        Modifies the member variable _d_reg_self_array with the
            gradient of itself wrt a reg parameter. Before an iteration,
            call the method reset_d_reg_self_array(self) to set
            everything in _d_reg_self_array to be zeros

        Args:
            index: time step to modify the array _d_reg_self_array
        """
        parameter_i = self[index]
        # for each reg parameter
        for key in self.keys():
            d_reg_self = self._d_reg_self_array[key]
            # add derivate of AR and MA terms (depends on past values)
            if "AR" in self.keys():
                d_reg_self[index] += self.arma.d_reg_ar_term(index, key)
            if "MA" in self.keys():
                d_reg_self[index] += self.arma.d_reg_ma_term(index, key)
            # add derivate of the systematic component
            if key == "reg":
                d_reg_self[index] += self.time_series.get_normalise_x(index)
            elif key == "AR":
                d_reg_self[index] += self.arma.ar_term(index)
            elif key == "MA":
                d_reg_self[index] += self.arma.ma_term(index)
            elif key == "const":
                d_reg_self[index] += 1
            # chain rule when differentiating exp()
            d_reg_self[index] *= parameter_i

    def d_reg_ar(self, index, key):
        """Derivate of the AR term at a time step

        \\partial \\Phi(i) / \\partial {parameter_key}

        Args:
            index: time step

        Returns:
            derivate of the AR term, scalar
        """
        grad_before = self._d_reg_self_array[key][index]
        grad = grad_before / self[index]
        if key == "const":
            grad -= 1
        return grad

    def d_reg_ma(self, index, key):
        """Derivate of the MA term at a time step

        \\partial \\Theta(i) / \\partial {parameter_key}
        Abstract method - subclasses to implement

        Args:
            index: time step
            key: name of the parameter to differentiate wrt (eg "reg", "AR",
                "MA", "const")

        Returns:
            derivate of the MA term, scalar
        """
        pass

    def get_reg_vector(self):
        """Return all regression parameters as a single vector

        Returns:
            regression parameter as a single vector. For ordering, see
                get_reg_vector_name()
        """
        parameter = np.empty(0)
        for value in self.values():
            parameter = np.concatenate((parameter, value))
        return parameter

    def get_reg_vector_name(self):
        """Return the name of each element of the regression parameter vector

        Return an array of strings with the format class_regression_parameter
            or if the regression parameter is "reg", the class_model_field
        """
        vector_name = []
        self_name = self.__class__.__name__
        for key, value in self.items():
            # put model field name
            if key == "reg":
                for i in range(len(value)):
                    vector_name.append(
                        self_name+"_"+self.time_series.model_field_name[i])
            # for AR and MA, name it as eg AR_1, MA_2
            elif key == "AR" or key == "MA":
                for i in range(len(value)):
                    vector_name.append(self_name+"_"+key+str(i+1))
            # else it is a constant
            else:
                vector_name.append(self_name+"_"+key)
        return vector_name

    def set_reg_vector(self, parameter):
        """Set all regression parameters using a single vector

        Args:
            parameter: regression parameter as a single vector. For ordering,
                see get_reg_vector_name()
        """
        dim_counter = 0
        for key in self.keys():
            n_dim_key = self[key].shape[0]
            self[key] = parameter[dim_counter:dim_counter+n_dim_key]
            dim_counter += n_dim_key

    def delete_array(self):
        """Delete arrays value_array and _d_reg_self_array
        """
        self.value_array = None
        self._d_reg_self_array = None

    def keys(self):
        """Return the keys of the regression parameters
        """
        return self.reg_parameters.keys()

    def values(self):
        """Return the regression parameters
        """
        return self.reg_parameters.values()

    def items(self):
        """Return the keys and regression parameters
        """
        return self.reg_parameters.items()

    def __len__(self):
        return len(self.value_array)

    def __str__(self):
        # return reg_parameters
        return self.reg_parameters.__str__()

    def __getitem__(self, key):
        # can return a value in value_array when key is a non-negative integer
        #
        # can return a value from the corresponding parameter in
        # time_series.fitted_time_series (if it exists) if key is a negative
        # integer
        #
        # can return a reg parameter when provided with "reg", "AR", "MA",
        # "const"
        if key in self.keys():
            return self.reg_parameters[key]
        else:
            # non-negative index, return value in value_array
            if key >= 0:
                return self.value_array[key]
            # negative index, return value from the past time series
            else:
                time_series_before = self.time_series.fitted_time_series
                # get the corresponding parameter from the past time series
                for parameter in time_series_before.cp_parameter_array:
                    if isinstance(parameter, self.__class__):
                        parameter_before = parameter
                        break
                return parameter_before[len(time_series_before) + key]

    def __setitem__(self, key, value):
        # can set a value in value_array when key is an integer
        #
        # can set a reg parameter when provided with "reg", "AR", "MA",
        # "const"
        if key in self.keys():
            self.reg_parameters[key] = value
        else:
            self.value_array[key] = value

    def __add__(self, other):
        # add reg parameters
        for key in self.keys():
            self[key] += other[key]
        return self

    def __mul__(self, other):
        # multiply reg parameters by a scalar
        for key in self.keys():
            self[key] *= other
        return self


class PoissonRate(Parameter):

    def __init__(self, n_model_field, n_arma):
        super().__init__(n_model_field, n_arma)

    def ma(self, y, z, poisson_rate, gamma_mean, gamma_dispersion):
        return (z - poisson_rate) / math.sqrt(poisson_rate)

    def d_self_ln_l(self, index):
        z = self.time_series.z_array[index]
        poisson_rate = self.value_array[index]
        return z/poisson_rate - 1

    def d_reg_ma(self, index, key):
        poisson_rate = self[index]
        z = self.time_series.z_array[index]
        return (-0.5*(z+poisson_rate) / math.pow(poisson_rate, 3/2)
                * self._d_reg_self_array[key][index])


class GammaMean(Parameter):

    def __init__(self, n_model_field, n_arma):
        super().__init__(n_model_field, n_arma)

    def ma(self, y, z, poisson_rate, gamma_mean, gamma_dispersion):
        if z > 0:
            return ((y - z * gamma_mean) / gamma_mean
                    / math.sqrt(gamma_dispersion * z))
        else:
            return 0.0

    def d_self_ln_l(self, index):
        y = self.time_series[index]
        z = self.time_series.z_array[index]
        mu = self[index]
        phi = self.time_series.gamma_dispersion[index]
        return (y-z*mu) / (phi*math.pow(mu, 2))

    def d_reg_ma(self, index, key):
        y = self.time_series[index]
        z = self.time_series.z_array[index]
        if z > 0:
            gamma_mean = self[index]
            gamma_dispersion = self.time_series.gamma_dispersion
            d_reg_gamma_mean = self._d_reg_self_array[key][index]
            if key in gamma_dispersion.keys():
                d_reg_gamma_dispersion = (gamma_dispersion
                                          ._d_reg_self_array[key][index])
            else:
                d_reg_gamma_dispersion = 0.0
            gamma_dispersion = gamma_dispersion[index]
            return (
                (
                    -(y * d_reg_gamma_mean)
                    - 0.5*(y-z*gamma_mean)*gamma_mean/gamma_dispersion
                    * d_reg_gamma_dispersion
                )
                / math.pow(gamma_mean, 2)*math.sqrt(z*gamma_dispersion)
            )
        else:
            return np.zeros_like(self[key])


class GammaDispersion(Parameter):

    def __init__(self, n_model_field):
        super().__init__(n_model_field, None)

    def d_self_ln_l(self, index):
        z = self.time_series.z_array[index]
        if z == 0:
            return 0
        else:
            y = self.time_series[index]
            mu = self.time_series.gamma_mean[index]
            phi = self[index]
            z_var = self.time_series.z_var_array[index]
            terms = np.zeros(7)
            terms[0] = z*math.log(mu)
            terms[1] = z*math.log(phi)
            terms[2] = -z*math.log(y)
            terms[3] = y/mu - z
            terms[4] = z*special.digamma(z/phi)
            terms[5] = z_var*special.polygamma(1, z/phi)/phi
            terms[6] = (0.5*z*z_var*special.polygamma(2, z/phi)
                        / math.pow(phi, 2))
            return np.sum(terms) / math.pow(phi, 2)
