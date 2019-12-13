import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import statsmodels.tsa.stattools as stats

from scipy.special import loggamma, digamma

class CompoundPoissonTimeSeries:
    """Compound Poisson Time Series with ARMA behaviour
    
    Initalise by passing initial parameters vai the constructor.
    The time series can be simulated
    The parameters can be estimated
    
    Attributes:
        x: design matrix of the model fields, shape (n, n_dim)
        n: length of time series
        n_dim: number of dimensions of the model fields
        poisson_rate_parameter: regressive and ARMA parameters (Parameters
            object)
        gamma_mean_parameter: regressive and ARMA parameters (Parameters object)
        gamma_dispersion_parameter: regressive and ARMA parameters (Parameters
            object)
        poisson_rate_array: poisson rate at each time step
        gamma_mean_array: gamma mean at each time step
        gamma_dispersion_array: gamma dispersion at each time step
        z_array: latent poisson variables at each time step
        y_array: compound poisson variables at each time step
        _tweedie_p_array: index parameter for each time step, a reparametrize
            using Tweedie
        _tweedie_phi_array: dispersion parameter for each time step, a
            reparametrize using Tweedie
    """
    
    _cp_sum_threshold = -37
    
    def __init__(self, x, poisson_rate_parameter,
                          gamma_mean_parameter,
                          gamma_dispersion_parameter):
        """
        Args:
            x: design matrix of the model fields, shape (n, n_dim)
            poisson_rate_parameter: Parameters object
            gamma_mean_parameter: Parameters object
            gamma_dispersion_parameter: Parameters object
        """
        self.x = x
        self.n = x.shape[0]
        self.n_dim = x.shape[1]
        self.poisson_rate_parameter = None
        self.gamma_mean_parameter = None
        self.gamma_dispersion_parameter = None
        self.poisson_rate_array = np.zeros(self.n)
        self.gamma_mean_array = np.zeros(self.n)
        self.gamma_dispersion_array = np.zeros(self.n)
        self.z_array = np.zeros(self.n)
        self.y_array = np.zeros(self.n)
        self._tweedie_p_array = np.zeros(self.n)
        self._tweedie_phi_array = np.zeros(self.n)
        
        self.set_parameters(poisson_rate_parameter, gamma_mean_parameter,
            gamma_dispersion_parameter)
    
    def set_parameters(self, poisson_rate_parameter, gamma_mean_parameter,
        gamma_dispersion_parameter):
        """Set the member variables of _parameter
        
        Set the member variables of _parameter, assign their parents and convert
            all parameters to numpy array
        
        Args:
            poisson_rate_parameter:
            gamma_mean_parameter:
            gamma_dispersion_parameter:
        """
        self.poisson_rate_parameter = poisson_rate_parameter
        self.gamma_mean_parameter = gamma_mean_parameter
        self.gamma_dispersion_parameter = gamma_dispersion_parameter
        #assign the parameters parents as self, this is so that the parameter
            #objects has access to the data 
        self.poisson_rate_parameter.assign_parent(self)
        self.gamma_mean_parameter.assign_parent(self)
        self.gamma_dispersion_parameter.assign_parent(self)
        #perpare the parameters by converting all to numpy array
        self.poisson_rate_parameter.convert_all_to_np()
        self.gamma_mean_parameter.convert_all_to_np()
        self.gamma_dispersion_parameter.convert_all_to_np()
    
    def set_observables(self, y_array):
        """Set the observed rainfall
        
        Args:
            y_array: rainfall for each day (vector)
        """
        self.y_array = y_array
    
    def simulate(self, rng):
        """Simulate a whole time series
        
        Simulate a time series given the model fields self.x, parameters
            and self.parameters. Modify the member variables poisson_rate_array,
            gamma_mean_array and gamma_dispersion_array with the parameters at
            each time step. Also modifies self.z_array and self.y_array with the
            simulated values.
        
        Args:
            rng: numpy.random.RandomState object
        """
        #simulate n times
        for i in range(self.n):
            #get the parameters of the compound Poisson at this time step
            self.update_poisson_gamma(i)
            #simulate this compound Poisson
            self.y_array[i], self.z_array[i] = simulate_cp(
                self.poisson_rate_array[i], self.gamma_mean_array[i],
                self.gamma_dispersion_array[i], rng)
    
    def update_poisson_gamma(self, index):
        """Update the variables of the Poisson and gamma random variables
        
        Updates the Poisson rate, gamma mean and gama dispersion parameters for
            a given time step and all the z before that time step. Modifies the
            member variables poisson_rate_array, gamma_mean_array and
            gamma_dispersion_array
        
        Args:
            index: the time step to update the parameters at
        """
        #regressive on the model fields and constant
        #exp link function, make it positive
        self.poisson_rate_array[index] = math.exp(
            np.dot(self.poisson_rate_parameter["reg"], self.x[index,:])
            + (self.poisson_rate_parameter["AR"]
                * self.poisson_rate_parameter.ar_term(index))
            + (self.poisson_rate_parameter["MA"]
                * self.poisson_rate_parameter.ma_term(index))
            + self.poisson_rate_parameter["const"])
        self.gamma_mean_array[index] = math.exp(
            np.dot(self.gamma_mean_parameter["reg"], self.x[index,:])
            + (self.gamma_mean_parameter["AR"]
                * self.gamma_mean_parameter.ar_term(index))
            + (self.gamma_mean_parameter["MA"]
                * self.gamma_mean_parameter.ma_term(index))
            + self.gamma_mean_parameter["const"])
        #dispersion does not have ARMA terms
        self.gamma_dispersion_array[index] = math.exp(
            np.dot(self.gamma_dispersion_parameter["reg"], self.x[index,:])
            + self.gamma_dispersion_parameter["const"])
    
    def e_step(self):
        """Does the E step of the EM algorithm
        
        Make estimates of the z and updates the poisson rate, gamma mean and
            gamma dispersion parameters for each time step. Modifies the member
            variables z_array, poisson_rate_array, gamma_mean_array,
            gamma_dispersion_array
        """
        #for each data point (forwards in time)
        for i in range(self.n):
            #update the parameter at this time step
            self.update_poisson_gamma(i)
            
            #get the parameters
            poisson_rate = self.poisson_rate_array[i]
            gamma_mean = self.gamma_mean_array[i]
            gamma_dispersion = self.gamma_dispersion_array[i]
            #save the Tweedie parameters
            p = (1+2*gamma_dispersion) / (1+gamma_dispersion)
            phi = (
                (gamma_dispersion+1) * math.pow(poisson_rate, 1-p)
                / math.pow(gamma_mean, p-2))
            self._tweedie_p_array[i] = p
            self._tweedie_phi_array[i] = phi
            
            #if the rainfall is zero, then z is zero (it has not rained)
            if self.y_array[i] == 0:
                self.z_array[i] = 0
            else:
                #work out the normalisation constant for the expectation
                normalisation_constant = self.ln_sum_w(i, 0)
                #work out the expectation
                self.z_array[i] = math.exp(
                    self.ln_sum_w(i, 1) - normalisation_constant)
    
    def z_max(self, index):
        """Gets the index of the biggest term in the compound Poisson sum
        
        Args:
            index: time step, y[index] must be positive
        
        Returns:
            positive integer, index of the biggest term in the compound Poisson
                sum
        """
        #get the optima with respect to the sum index, then round it to get an
            #integer
        y = self.y_array[index]
        p = self._tweedie_p_array[index]
        phi = self._tweedie_phi_array[index]
        z_max = round(math.exp((2-p)*math.log(y)-math.log(phi)-math.log(2-p)))
        #if the integer is 0, then set the index to 1
        if z_max == 0:
            z_max = 1
        return z_max
    
    def ln_wz(self, index, z):
        """Return a log term from the compound Poisson sum
        
        Args:
            index: time step, y[index] must be positive
            z: Poisson variable or index of the sum element
        
        Returns:
            log compopund Poisson term
        """
        
        #declare array of terms to be summed to work out ln_wz
        terms = np.zeros(6)
        #retrieve variables
        y = self.y_array[index]
        alpha = 1/self.gamma_dispersion_array[index]
        p = self._tweedie_p_array[index]
        phi = self._tweedie_phi_array[index]

        #work out each individual term
        terms[0] = -z*alpha*math.log(p-1)
        terms[1] = z*alpha*math.log(y)
        terms[2] = -z*(1+alpha)*math.log(phi)
        terms[3] = -z*math.log(2-p)
        terms[4] = -loggamma(1+z)
        terms[5] = -loggamma(alpha*z)
        #sum the terms to get the log compound Poisson sum term
        ln_wz = np.sum(terms)
        return ln_wz
    
    def ln_sum_w(self, index, z_pow):
        """Works out the compound Poisson sum, only important terms are summed
        
        Args:
            index: time step, y[index] must be positive
            z_pow: 0,1,2,..., used for taking the sum for y^yPow * Wy which is
                used for taking expectations
        
        Returns:
            log compound Poisson sum
        """
        
        y = self.y_array[index]
        
        #get the y with the biggest term in the compound Poisson sum
        z_max = self.z_max(index)
        #get the biggest log compound Poisson term + any expectation terms
        ln_w_max = self.ln_wz(index, z_max) + z_pow*math.log(z_max)
        
        #declare array of compound poisson terms
        #each term is a ratio of the compound poisson term with the maximum 
            #compound poisson term
        #the first term is 1, that is exp[ln(W_ymax)-ln(W_ymax)] = 1;
        terms = [1]
        
        #declare booleans is_got_z_l and is_got_z_u
        #these are true if we got the lower and upper bound respectively for
            #the compound Poisson sum
        is_got_z_l = False
        is_got_z_u = False
        
        #declare the summation bounds, z_l for the lower bound, z_u for the
            #upper bound
        z_l = z_max
        z_u = z_max
        
        #calculate the compound poisson terms starting at yL and working
            #downwards if the lower bound is 1, can't go any lower and set
            #is_got_z_l to be true
        if z_l == 1:
            is_got_z_l = True
        
        #while we haven't got a lower bound
        while not is_got_z_l:
            #lower the lower bound
            z_l -= 1
            #if the lower bound is 0, then set is_got_z_l to be true and raise
                #the lower bound back by one
            if z_l == 0:
                is_got_z_l = True
                z_l += 1
            else: #else the lower bound is not 0
                #calculate the log ratio of the compound poisson term with the
                    #maximum compound poisson term
                log_ratio = np.sum(
                    [self.ln_wz(index, z_l), z_pow*math.log(z_l), -ln_w_max])
                #if this log ratio is bigger than the threshold
                if log_ratio > CompoundPoissonTimeSeries._cp_sum_threshold:
                    #append the ratio to the array of terms
                    terms.append(math.exp(log_ratio))
                else:
                    #else the log ratio is smaller than the threshold
                    #set is_got_z_l to be true and raise the lower bound by 1
                    is_got_z_l = True
                    z_l += 1

        #while we haven't got an upper bound
        while not is_got_z_u:
            #raise the upper bound by 1
            z_u += 1;
            #calculate the log ratio of the compound poisson term with the
                #maximum compound poisson term
            log_ratio = np.sum(
                [self.ln_wz(index, z_u), z_pow*math.log(z_u), -ln_w_max])
            #if this log ratio is bigger than the threshold
            if log_ratio > CompoundPoissonTimeSeries._cp_sum_threshold:
                #append the ratio to the array of terms
                terms.append(math.exp(log_ratio))
            else:
                #else the log ratio is smaller than the threshold
                #set is_got_z_u to be true and lower the upper bound by 1
                is_got_z_u = True
                z_u -= 1

        #work out the compound Poisson sum 
        ln_sum_w = ln_w_max + math.log(np.sum(terms))
        return ln_sum_w
    
    def m_step(self):
        """Does the M step of the EM algorithm
        
        Estimates the parameters given the observed rainfall y and the latent
            variable z using gradient descent. The objective function is the log
            likelihood assuming the latent variables are observed
        """
        #set their member variables containing the gradients to be zero, they
            #are calculated in the next step
        self.poisson_rate_parameter.reset_d_parameter_self_array()
        self.gamma_mean_parameter.reset_d_parameter_self_array()
        self.gamma_dispersion_parameter.reset_d_parameter_self_array()
        #for each time step, work out d itself / d parameter where itself can be
            #poisson_rate or gamma_mean for example and parameter can be the AR
            #or MA parameters
        #gradient at time step i depends on gradient at time step i-1, therefore
            #use loop over range(n)
        for i in range(self.n):
            self.update_poisson_gamma(i)
            self.poisson_rate_parameter.calculate_d_parameter_self_i(i)
            self.gamma_mean_parameter.calculate_d_parameter_self_i(i)
            self.gamma_dispersion_parameter.calculate_d_parameter_self_i(i)
        #get the gradient of the log likelihood
        grad_poisson_rate = self.poisson_rate_parameter.d_parameter_ln_l()
        grad_gamma_mean = self.gamma_mean_parameter.d_parameter_ln_l()
        grad_gamma_dispersion = (
            self.gamma_dispersion_parameter.d_parameter_ln_l())
        #multiply the gradient by a factor to be used for gradient descent
        grad_poisson_rate *= 0.1/self.n
        grad_gamma_mean *= 0.1/self.n
        grad_gamma_dispersion *= 0.1/self.n
        #do gradient descent
        self.poisson_rate_parameter += grad_poisson_rate
        self.gamma_mean_parameter += grad_gamma_mean
        self.gamma_dispersion_parameter += grad_gamma_dispersion
    
    class Parameters:
        """Contains regressive, autoregressive, moving average and constant
            parameters for the compound Poisson time series model
        
        Attributes:
            parameters: dictionary with keys "reg", "AR", "MA", "const" with
                corresponding values regression parameters
            _parent: CompoundPoissonTimeSeries object containing this
            _d_parameter_self_array: array of derivates of itself wrt parameter
        """
        
        def __init__(self, n_dim):
            """
            Args:
                n_dim: number of dimensions
            """
            self.n_dim = n_dim
            self.parameters = {
                "reg": np.zeros(n_dim),
                "const": 0.0,
            }
            self._parent = None
            self._self_array = None
            self._d_parameter_self_array = None
        
        def convert_all_to_np(self):
            """Convert all values in self.parameters to be numpy array
            """
            for key, value in self.parameters.items():
                self.parameters[key] = np.asarray(value)
                if key == "reg":
                    self.parameters[key] = np.reshape(
                        self.parameters[key], self._parent.n_dim)
                else:
                    self.parameters[key] = np.reshape(self.parameters[key], 1)
        
        def d_self_ln_l(self):
            """Derivate of the log likelihood wrt itself for each time step
            
            Returns:
                vector of gradients
            """
            pass
        
        def copy(self):
            """Return deep copy of itself
            """
            copy = Parameters(self.n_dim)
            for key, value in self.parameters.items():
                if key == "reg":
                    copy.parameters[key] = value.copy()
                else:
                    copy.parameters[key] = value
            return copy
        
        def assign_parent(self, parent):
            """Assign parent
            
            Assign the member variable _parent which points to the
                CompoundPoissonTimeSeries object with owns self
            """
            self._parent = parent
        
        def ar_term(self, index):
            """AR term at a time step
            
            Returns the autoregressive term, log(self[index-1]) - constant term
            
            Args:
                index: time step
            
            Returns:
                the AR term at a time step
            """
            if index > 0:
                return (math.log(self._self_array[index-1])
                    - self.parameters["const"])
            else:
                return 0.0
        
        def ma_term(self, index):
            """MA term at a time step
            
            Returns the moving average term, (self[index-1] -
                expected(self[index-1])) / std(self[index-1])
            
            Args:
                index: time step
            
            Returns:
                the MA term at a time step
            """
            pass
        
        def d_parameter_ma(self, index, key):
            """Derivate of the MA term at a time step
            
            Args:
                index: time e_step
                key: name of the parameter to derivate wrt
            
            Returns:
                derivate of the MA term
            
            """
            pass
        
        def reset_d_parameter_self_array(self):
            """Reset _d_parameter_self_array
            
            Set all values in _d_parameter_self_array to be numpy zeros
            
            """
            self._d_parameter_self_array = {}
            keys = self.parameters.keys()
            for key, value in self.parameters.items():
                self._d_parameter_self_array[key] = np.zeros(
                    (self._parent.n, value.size))
        
        def calculate_d_parameter_self_i(self, index):
            """Calculates the derivate of itself wrt parameter
            
            Modifies the member variable _d_parameter_self_array
            """
            parameter_i = self._self_array[index]
            keys = self._d_parameter_self_array.keys()
            if index > 0:
                parameter_i_1 = self._self_array[index-1]
            for key in self.parameters.keys():
                d_parameter_self = self._d_parameter_self_array[key]
                if index > 0:
                    #AR and MA terms
                    if "AR" in keys:
                        d_parameter_self[index] += (
                            self.parameters["AR"] * d_parameter_self[index-1]
                            / parameter_i_1)
                    if "MA" in keys:
                        d_parameter_self[index] += (
                            self.parameters["MA"]
                            * self.d_parameter_ma(index, key))
                if key == "reg":
                    d_parameter_self[index] +=  self._parent.x[index,:]
                elif key == "AR":
                    d_parameter_self[index] += self.ar_term(index)
                elif key == "MA":
                    d_parameter_self[index] += self.ma_term(index)
                elif key == "const":
                    d_parameter_self[index] += 1
                    if "AR" in keys:
                        d_parameter_self[index] -= self.parameters["AR"]
                d_parameter_self[index] *= parameter_i
        
        def d_parameter_ln_l(self):
            """The derivate of the log likelihood wrt parameters
            """
            d_self_ln_l = self.d_self_ln_l()
            for key in self._d_parameter_self_array.keys():
                for i in range(self._parent.n):
                    self._d_parameter_self_array[key][i] *= d_self_ln_l[i]
                value = self._d_parameter_self_array[key]
                if value.ndim == 1:
                    self._d_parameter_self_array[key] = np.sum(value)
                else:
                    self._d_parameter_self_array[key] = np.sum(value, 0)
            gradient = CompoundPoissonTimeSeries.Parameters(self._parent.n_dim)
            gradient.parameters = self._d_parameter_self_array
            return gradient
        
        def __str__(self):
            return self.parameters.__str__()
        
        def __getitem__(self, key):
            return self.parameters[key]
        
        def __setitem__(self, key, value):
            self.parameters[key] = value
        
        def __iadd__(self, other):
            for key in self.parameters.keys():
                self.parameters[key] += other.parameters[key]
            return self
        
        def __imul__(self, other):
            for key in self.parameters.keys():
                self.parameters[key] *= other
            return self

    class PoissonRateParameters(Parameters):
        def __init__(self, n_dim):
            super().__init__(n_dim)
            self.parameters["AR"] = 0.0
            self.parameters["MA"] = 0.0
        def assign_parent(self, parent):
            super().assign_parent(parent)
            self._self_array = parent.poisson_rate_array
        def ma_term(self, index):
            if index > 0:
                return (
                    (self._parent.z_array[index-1] - self._self_array[index-1])
                    / math.sqrt(self._self_array[index-1]))
            else:
                return 0.0
        def d_self_ln_l(self):
            z = self._parent.z_array
            poisson_rate = self._parent.poisson_rate_array
            return z/poisson_rate - 1
        def d_parameter_ma(self, index, key):
            poisson_rate_before = self._parent.poisson_rate_array[index-1]
            return (-0.5*math.sqrt(poisson_rate_before)
                *(1+self._parent.z_array[index-1]/poisson_rate_before)
                *self._d_parameter_self_array[key][index-1])

    class GammaMeanParameters(Parameters):
        def __init__(self, n_dim):
            super().__init__(n_dim)
            self.parameters["AR"] = 0.0
            self.parameters["MA"] = 0.0
        def assign_parent(self, parent):
            super().assign_parent(parent)
            self._self_array = parent.gamma_mean_array
        def ma_term(self, index):
            if index > 0 and self._parent.z_array[index-1] > 0:
                return (
                    (self._parent.y_array[index-1]/self._parent.z_array[index-1]
                        - self._parent.gamma_mean_array[index-1])
                    / self._parent.gamma_mean_array[index-1]
                    / math.sqrt(
                        self._parent.gamma_dispersion_array[index-1]
                        * self._parent.z_array[index-1]))
            else:
                return 0.0
        def d_self_ln_l(self):
            y = self._parent.y_array
            z = self._parent.z_array
            mu = self._self_array
            phi = self._parent.gamma_dispersion_array
            return (y-z*mu) / (phi*mu*mu)
        def d_parameter_ma(self, index, key):
            y = self._parent.y_array[index-1]
            z = self._parent.z_array[index-1]
            if z > 0:
                gamma_mean = self._self_array[index-1]
                gamma_dispersion = self._parent.gamma_dispersion_array[index-1]
                d_parameter_gamma_mean = (
                    self._d_parameter_self_array[key][index-1])
                gamma_dispersion_parameter = (self
                    ._parent.gamma_dispersion_parameter
                    ._d_parameter_self_array)
                if key in gamma_dispersion_parameter.keys():
                    d_parameter_gamma_dispersion = (self
                        ._parent.gamma_dispersion_parameter
                        ._d_parameter_self_array[key][index-1])
                else:
                    d_parameter_gamma_dispersion = 0.0
                return ( ( -(y * d_parameter_gamma_mean)
                        - 0.5*(y-z*gamma_mean)*gamma_mean
                        *d_parameter_gamma_dispersion
                        / gamma_dispersion
                    )
                    / ( math.pow(gamma_mean,2) * math.pow(z,3/2)
                        * math.sqrt(gamma_dispersion)))
            else:
                return 0.0

    class GammaDispersionParameters(Parameters):
        def __init__(self, n_dim):
            super().__init__(n_dim)
        def assign_parent(self, parent):
            super().assign_parent(parent)
            self._self_array = parent.gamma_dispersion_array
        def d_self_ln_l(self):
            d_self_ln_l = np.zeros(self._parent.n)
            for i in range(self._parent.n):
                z = self._parent.z_array[i]
                if z == 0:
                    d_self_ln_l[i] = 0
                else:
                    y = self._parent.y_array[i]
                    mu = self._self_array[i]
                    phi = self._parent.gamma_dispersion_array[i]
                    d_self_ln_l[i] = (z*(math.log(mu*phi/y)-1) + y/mu
                        + digamma(z/phi)) / math.pow(phi,2)
            return d_self_ln_l

def simulate_cp(poisson_rate, gamma_mean, gamma_dispersion, rng):
    """Simulate a single compound poisson random variable
    
    Args:
        poisson_rate: scalar
        gamma_mean: scalar
        gamma_dispersion: scalar
        rng: numpy.random.RandomState object
    
    Returns:
        tuple contain vectors of y (compound Poisson random variable) and z
            (latent Poisson random variable)
    """
    
    z = rng.poisson(poisson_rate) #poisson random variable
    shape = z / gamma_dispersion #gamma shape parameter
    #if z is zero, set the parameters of the gamma distribution to be zero
    if z > 0:
        scale = gamma_mean/shape
    else:
        scale = 0
    y = rng.gamma(shape, scale=scale) #gamma random variable
    return (y, z)

def main():
    
    rng = random.RandomState(np.uint32(372625178))
    n = 10000
    x = np.zeros((n,1))
    for i in range(n):
        x[i] = 0.8*math.sin(2*math.pi/365 * i) + rng.normal()
    
    poisson_rate_parameters = CompoundPoissonTimeSeries.PoissonRateParameters(1)
    poisson_rate_parameters["reg"] = 0.098
    poisson_rate_parameters["AR"] = 0.713
    poisson_rate_parameters["MA"] = 0.11
    poisson_rate_parameters["const"] = 0.355
    gamma_mean_parameters = CompoundPoissonTimeSeries.GammaMeanParameters(1)
    gamma_mean_parameters["reg"] = 0.066
    gamma_mean_parameters["AR"] = 0.4
    gamma_mean_parameters["MA"] = 0.24
    gamma_mean_parameters["const"] = 1.3
    gamma_dispersion_parameters = (
        CompoundPoissonTimeSeries.GammaDispersionParameters(1))
    gamma_dispersion_parameters["reg"] = 0.07
    gamma_dispersion_parameters["const"] = 0.373
    
    compound_poisson_time_series = CompoundPoissonTimeSeries(
        x, poisson_rate_parameters, gamma_mean_parameters,
        gamma_dispersion_parameters)
    compound_poisson_time_series.simulate(rng)
    
    compound_poisson_time_series.set_parameters(
        CompoundPoissonTimeSeries.PoissonRateParameters(1),
        CompoundPoissonTimeSeries.GammaMeanParameters(1),
        CompoundPoissonTimeSeries.GammaDispersionParameters(1))
    
    for i in range(10):
        print(i)
        compound_poisson_time_series.e_step()
        compound_poisson_time_series.m_step()
    compound_poisson_time_series.e_step()
    print(compound_poisson_time_series.poisson_rate_parameter)
    print(compound_poisson_time_series.gamma_mean_parameter)
    print(compound_poisson_time_series.gamma_dispersion_parameter)
    
    compound_poisson_time_series.simulate(rng)
    
    y = compound_poisson_time_series.y_array
    poisson_rate_array = compound_poisson_time_series.poisson_rate_array
    gamma_mean_array = compound_poisson_time_series.gamma_mean_array
    gamma_dispersion_array = compound_poisson_time_series.gamma_dispersion_array
    
    acf = stats.acf(y, nlags=100, fft=True)
    pacf = stats.pacf(y, nlags=10)
    
    plt.figure()
    plt.plot(y)
    plt.xlabel("Time (day)")
    plt.ylabel("Rainfall (mm)")
    plt.savefig("../figures/simulation_rain.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(x)
    plt.xlabel("Time (day)")
    plt.ylabel("Model field")
    plt.savefig("../figures/simulation_model_field.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.bar(np.asarray(range(acf.size)), acf)
    plt.xlabel("Time (day)")
    plt.ylabel("Autocorrelation")
    plt.savefig("../figures/simulation_acf.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.bar(np.asarray(range(pacf.size)), pacf)
    plt.xlabel("Time (day)")
    plt.ylabel("Partial autocorrelation")
    plt.savefig("../figures/simulation_pacf.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(poisson_rate_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Poisson rate")
    plt.savefig("../figures/simulation_lambda.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(gamma_mean_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Mean of gamma")
    plt.savefig("../figures/simulation_mu.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(gamma_dispersion_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Dispersion")
    plt.savefig("../figures/simulation_dispersion.png")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(range(n),compound_poisson_time_series.z_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Z")
    plt.savefig("../figures/simulation_z.png")
    plt.show()
    plt.close()
    
main()
