import math
import numpy as np

from Arma import Arma
from scipy.special import digamma, polygamma

class Parameter:
    """Dynamic parameters of the compound Poisson
    
    An abstract class
    Contains the parameter at each time step as it envolves. Objects from
        this class also contains regressive parameters, such as regressive,
        autoregressive, moving average and constant for the compound
        Poisson time series model
    Values of itself can be obtained or set using self[i] where i is an
        integer representing the time step
    Reg parameters can be obtained or set using self[key] where key can be
        "reg", "AR", "MA" or "const"
    
    This is an abstract class with the following methods need to be
        implemented:
        -ma_term(self, index)
        -d_self_ln_l(self)
        -d_parameter_ma(self, index, key)
    
    Attributes:
        n: number of time steps
        n_dim: number of dimensions the model fields has
        reg_parameters: dictionary containing regressive (reg) parameters
            with keys "reg", "AR", "MA", "const" which corresponds to the
            names of the reg parameters
        arma: object to evalute arma terms
        value_array: array of values of the parameter for each time step
        _parent: TimeSeries object containing this
        _d_reg_self_array: dictionary containing arrays of derivates
            of itself wrt reg parameters
    """
    
    def __init__(self, n_dim):
        """
        Args:
            n_dim: number of dimensions
        """
        self.n = None
        self.n_dim = n_dim
        self.reg_parameters = {
            "reg": np.zeros(n_dim),
            "const": 0.0,
        }
        self.arma = None
        self.value_array = None
        self._parent = None
        self._d_reg_self_array = None
    
    def assign_parent(self, parent):
        """Assign parent
        
        Assign the member variable _parent which points to the
            TimeSeries object which owns self
        """
        self._parent = parent
        self.n = parent.n
        self.value_array = np.zeros(self.n)
        self.arma = Arma(self)
    
    def copy(self):
        """Return deep copy of itself
        """
        copy = self.copy_reg()
        copy.assign_parent(self._parent)
        copy.value_array = self.value_array.copy()
        return copy
    
    def copy_reg(self):
        """Return deep copy of the regression parameters
        """
        copy = self.__class__(self.n_dim)
        for key, value in self.reg_parameters.items():
            if type(value) is np.ndarray:
                copy[key] = value.copy()
            else:
                copy[key] = value
        return copy
    
    def convert_all_to_np(self):
        """Convert all values in self.reg_parameters to be numpy array
        """
        for key, value in self.reg_parameters.items():
            self[key] = np.asarray(value)
            if key == "reg":
                self[key] = np.reshape(self[key], self.n_dim)
            else:
                self[key] = np.reshape(self[key], 1)
    
    def update_value_array(self, index):
        """Update the variables of value_array
        
        Updates and modifies the value_array at a given time step given all
            the z before that time step.
        
        Args:
            index: the time step to update the parameters at
        """
        #regressive on the model fields and constant
        exponent = 0.0
        exponent += self["const"]
        exponent += np.dot(self["reg"], self._parent.get_normalise_x(index))
        if "AR" in self.keys():
            exponent += self["AR"] * self.ar_term(index)
        if "MA" in self.keys():
            exponent += self["MA"] * self.ma_term(index)
        #exp link function, make it positive
        self[index] = math.exp(exponent)
    
    def cast_arma(self, arma_class):
        """Cast the arma object
        
        Update the member variable arma to be of another type using a provided
            class
        
        Args:
            arma_class: class object, self will be passed into the constructor
        """
        self.arma = arma_class(self)
    
    def ar_term(self, index):
        """AR term at a time step
        
        Returns the AR term at a given time step. Uses the arma object.
        """
        return self.arma.ar(index)
    
    def ma_term(self, index):
        """MA term at a time step
        
        Returns the MA term at a given time step. Uses the arma object.
        """
        return self.arma.ma(index)
    
    def ar(self, parameter):
        """AR term given parameters
        
        Returns the autoregressive term given parameters, to be used by the arma
            object.
        """
        return math.log(parameter) - self["const"]
    
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
            called for all index. See the method m_step in TimeSeries
        
        Args:
            step_size: change parameters by step_size/self.n * gradient
        """
        gradient = self.d_reg_ln_l()
        gradient *= step_size/self.n
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
        
        Returns a dictionary of the derivate of the log likelihood wrt reg
            parameters. Requires the following methods to be called
            beforehand:
            -reset_d_reg_self_array()
            -calculate_d_reg_self_i(index) where the outer loop is
                for each time step and the inner loop is for each reg
                parameter. This is required because the derivate of
                GammaMean.ma_term[index] depends on
                GammaDispersion._d_reg_self_array[index-1]. As a result, the
                derivate of all reg parameters must be found together for
                each time step
            See the method TimeSeries.m_step(self) to see
            example on how to call this method
        
        Returns:
            dictionary of the derivate of the log likelihood wrt reg
                parameters with keys "reg", "const" as well as if available
                "AR" and "MA"
        """
        d_self_ln_l = self.d_self_ln_l_all()
        d_reg_ln_l = self._d_reg_self_array
        for key in self.keys():
            for i in range(self.n):
                #chain rule
                d_reg_ln_l[key][i] = (
                    self._d_reg_self_array[key][i] * d_self_ln_l[i])
            #sum over all time steps (chain rule for partial diff)
            value = d_reg_ln_l[key]
            if value.ndim == 1:
                d_reg_ln_l[key] = np.sum(value)
            else:
                d_reg_ln_l[key] = np.sum(value, 0)
        #put the gradient in a Parameter object and return it
        gradient = Parameter(self.n_dim)
        gradient.reg_parameters = d_reg_ln_l
        return gradient
    
    def d_self_ln_l_all(self):
        """Derivate of the log likelihood wrt itself for all time steps
        
        Seethe method d_reg_ln_l() for any requirements
        
        Returns:
            vector of gradients
        """
        d_self_ln_l = np.zeros(self.n)
        for i in range(self.n):
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
            self._d_reg_self_array[key] = np.zeros(
                (self.n, value.size))
    
    def calculate_d_reg_self_i(self, index):
        """Calculates the derivate of itself wrt parameter
        
        Modifies the member variable _d_reg_self_array with the
            gradient of itself wrt a reg parameter. Before an iteration,
            call the method reset_d_reg_self_array(self) to set
            everything in _d_reg_self_array to be zeros
        
        Args:
            index: time step to modify the array _d_reg_self_array
        """
        parameter_i = self[index]
        for key in self.keys():
            d_reg_self = self._d_reg_self_array[key]
            if index > 0:
                #AR and MA terms
                parameter_i_1 = self[index-1]
                if "AR" in self.keys():
                    d_reg_self[index] += (self["AR"]
                        * d_reg_self[index-1] / parameter_i_1)
                if "MA" in self.keys():
                    d_reg_self[index] += (self["MA"]
                        * self.d_parameter_ma(index, key))
            if key == "reg":
                d_reg_self[index] += self._parent.get_normalise_x(index)
            elif key == "AR":
                d_reg_self[index] += self.ar_term(index)
            elif key == "MA":
                d_reg_self[index] += self.ma_term(index)
            elif key == "const":
                d_reg_self[index] += 1
                if "AR" in self.keys() and index > 0:
                    d_reg_self[index] -= self["AR"]
            d_reg_self[index] *= parameter_i
    
    def d_parameter_ma(self, index, key):
        """Derivate of the MA term at a time step
        
        Abstract method - subclasses to implement
        
        Args:
            index: time step
            key: name of the parameter to derivate wrt
        
        Returns:
            derivate of the MA term
        
        """
        pass
    
    def get_reg_vector(self):
        """Return all regression parameters as a single vector
        
        Returns:
            regression parameter as a single vector
        """
        parameter = np.empty(0)
        for value in self.values():
            parameter = np.concatenate((parameter, value))
        return parameter
    
    def get_reg_vector_name(self):
        """Return the name of each element of the regression parameter vector
        
        Return an array of strings with the format class_regression_parameter or
            if the regression parameter is "reg", the class_model_field
        """
        vector_name = []
        self_name = self.__class__.__name__ 
        for key, value in self.items():
            if key == "reg":
                for i in range(self.n_dim):
                    vector_name.append(
                        self_name+"_"+self._parent.model_field_name[i])
            else:
                vector_name.append(self_name+"_"+key)
        return vector_name
    
    def set_reg_vector(self, parameter):
        """Set all regression parameters using a single vectors
        
        Args:
            parameter: regression parameter as a single vector
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
    
    def __str__(self):
        #print reg_parameters
        return self.reg_parameters.__str__()
    
    def __getitem__(self, key):
        #can return a value in value_array when key is a non-negative integer
        #can return a value from the corresponding parameter in
            #_parent.fitted_time_series (if it exists) if key is a negative
            #integer
        #can return a reg parameter when provided with "reg", "AR", "MA",
            #"const"
        if key in self.keys():
            return self.reg_parameters[key]
        else:
            #non-negative index, return value in value_array
            if key >= 0:
                return self.value_array[key]
            #negative index, return value from the past time series
            else:
                time_series_before = self._parent.fitted_time_series
                #get the corresponding parameter from the past time series
                for parameter in time_series_before.cp_parameter_array:
                    if isinstance(parameter, self.__class__):
                        parameter_before = parameter
                        break
                return parameter_before[parameter_before.n + key]
    
    def __setitem__(self, key, value):
        #can set a value in value_array when key is an integer
        #can set a reg parameter when provided with "reg", "AR", "MA",
            #"const"
        if key in self.keys():
            self.reg_parameters[key] = value
        else:
            self.value_array[key] = value
    
    def __add__(self, other):
        #add reg parameters
        for key in self.keys():
            self[key] += other[key]
        return self
    
    def __mul__(self, other):
        #multiply reg parameters by a scalar
        for key in self.keys():
            self[key] *= other
        return self

class PoissonRate(Parameter):
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.reg_parameters["AR"] = 0.0
        self.reg_parameters["MA"] = 0.0
    def ma(self, y, z, poisson_rate, gamma_mean, gamma_dispersion):
        return (z - poisson_rate) / math.sqrt(poisson_rate)
    def d_self_ln_l(self, index):
        z = self._parent.z_array[index]
        poisson_rate = self.value_array[index]
        return z/poisson_rate - 1
    def d_parameter_ma(self, index, key):
        poisson_rate = self[index-1]
        z = self._parent.z_array[index-1]
        return (-0.5*(z+poisson_rate) / math.pow(poisson_rate, 3/2)
            *self._d_reg_self_array[key][index-1])

class GammaMean(Parameter):
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.reg_parameters["AR"] = 0.0
        self.reg_parameters["MA"] = 0.0
    def ma(self, y, z, poisson_rate, gamma_mean, gamma_dispersion):
        if z > 0:
            return ((y - z* gamma_mean) / gamma_mean
                / math.sqrt(gamma_dispersion * z))
        else:
            return 0.0
    def d_self_ln_l(self, index):
        y = self._parent.y_array[index]
        z = self._parent.z_array[index]
        mu = self[index]
        phi = self._parent.gamma_dispersion[index]
        return (y-z*mu) / (phi*math.pow(mu,2))
    def d_parameter_ma(self, index, key):
        y = self._parent.y_array[index-1]
        z = self._parent.z_array[index-1]
        if z > 0:
            gamma_mean = self[index-1]
            gamma_dispersion = self._parent.gamma_dispersion
            d_reg_gamma_mean = self._d_reg_self_array[key][index-1]
            if key in gamma_dispersion.keys():
                d_reg_gamma_dispersion = (gamma_dispersion
                    ._d_reg_self_array[key][index-1])
            else:
                d_reg_gamma_dispersion = 0.0
            gamma_dispersion = gamma_dispersion[index-1]
            return (
                (
                    -(y * d_reg_gamma_mean)
                    - 0.5*(y-z*gamma_mean)*gamma_mean/gamma_dispersion
                    * d_reg_gamma_dispersion
                )
                /
                (
                    math.pow(gamma_mean,2)*math.sqrt(z*gamma_dispersion)
                )
            )
        else:
            return 0.0

class GammaDispersion(Parameter):
    def __init__(self, n_dim):
        super().__init__(n_dim)
    def d_self_ln_l(self, index):
        z = self._parent.z_array[index]
        if z == 0:
            return 0
        else:
            y = self._parent.y_array[index]
            mu = self._parent.gamma_mean[index]
            phi = self[index]
            z_var = self._parent.z_var_array[index]
            terms = np.zeros(7)
            terms[0] = z*math.log(mu)
            terms[1] = z*math.log(phi)
            terms[2] = -z*math.log(y)
            terms[3] = y/mu - z
            terms[4] = z*digamma(z/phi)
            terms[5] = z_var*polygamma(1,z/phi)/phi
            terms[6] = 0.5*z*z_var*polygamma(2,z/phi)/ math.pow(phi,2)
            return np.sum(terms) / math.pow(phi,2)
