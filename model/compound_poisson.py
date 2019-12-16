import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import statsmodels.tsa.stattools as stats

from scipy.special import loggamma, digamma

class CompoundPoissonTimeSeries:
    """Compound Poisson Time Series with ARMA behaviour
    
    A time series distribued as compound Poisson with dynamic varying
        parameters. Objects from this class can simulate the model and fit the
        model.
    Initalise the model by passing initial parameters via the constructor. This
        is used to simulate or initalise the EM algorithm.
    There is a simuate method
    
    Attributes:
        x: design matrix of the model fields, shape (n, n_dim)
        n: length of time series
        n_dim: number of dimensions of the model fields
        poisson_rate: PoissonRate object containing the parameter at each time
            step and parameters for the regressive and ARMA parameters elements
        gamma_mean:  GammaMean containing the parameter at each time step and
            parameters for the regressive and ARMA parameters elements
        gamma_dispersion:  GammaDispersion containing the parameter at each time 
            step and parameters for the regressive and ARMA parameters elements
        cp_parameter_array: dictionary pointing to poisson_rate, gamma_mean and
            gamma_dispersion
        z_array: latent poisson variables at each time step
        y_array: compound poisson variables at each time step
        _tweedie_p_array: index parameter for each time step, a reparametrize
            using Tweedie
        _tweedie_phi_array: dispersion parameter for each time step, a
            reparametrize using Tweedie
    """
    
    _cp_sum_threshold = -37
    _step_size = 0.1
    _n_em = 100
    _min_ln_l_ratio = 0.0001
    
    def __init__(self, x, poisson_rate, gamma_mean, gamma_dispersion):
        """
        Args:
            x: design matrix of the model fields, shape (n, n_dim)
            poisson_rate: PoissonRate object
            gamma_mean: GammaMean object
            gamma_dispersion: GammaDispersion object
        """
        self.x = x
        self.n = x.shape[0]
        self.n_dim = x.shape[1]
        self.poisson_rate = None
        self.gamma_mean = None
        self.gamma_dispersion = None
        #dictionary containing poisson_rate, gamma_mean and gamma_dispersion
        self.cp_parameter_array = None
        self.z_array = np.zeros(self.n)
        self.y_array = np.zeros(self.n)
        #array of compound poisson parameters in Tweedie form
        self._tweedie_phi_array = np.zeros(self.n)
        self._tweedie_p_array = np.zeros(self.n)
        
        self.set_parameters(poisson_rate, gamma_mean, gamma_dispersion)
    
    def set_parameters(self, poisson_rate, gamma_mean, gamma_dispersion):
        """Set the member variables of the cp paramters
        
        Set the member variables of poisson_rate, gamma_mean,
            gamma_dispersion and cp_parameter_array. Itself is assigned as the
            parent to these objects. The reg parameters in these objects are
            converted to numpy array
        
        Args:
            poisson_rate: PoissonRate object
            gamma_mean: GammaMean object
            gamma_dispersion: GammaDispersion object
        """
        self.poisson_rate = poisson_rate
        self.gamma_mean = gamma_mean
        self.gamma_dispersion = gamma_dispersion
        #make dictionary of these variables
        self.cp_parameter_array = {
            "poisson_rate": poisson_rate,
            "gamma_mean": gamma_mean,
            "gamma_dispersion": gamma_dispersion
        }
        for parameter in self.cp_parameter_array.values():
            #assign the parameters parents as self, this is so that the
                #parameter objects has access to all member variables and
                #methods
            parameter.assign_parent(self)
            #prepare the parameters by converting all to numpy array
            parameter.convert_all_to_np()
    
    def set_observables(self, y_array):
        """Set the observed rainfall
        
        Args:
            y_array: rainfall for each day (vector)
        """
        self.y_array = y_array
    
    def simulate(self, rng):
        """Simulate a whole time series
        
        Simulate a time series given the model fields self.x and cp parameters.
            Modify the member variables poisson_rate, gamma_mean and
            gamma_dispersion by updating its values at each time step.
            Also modifies self.z_array and self.y_array with the
            simulated values.
        
        Args:
            rng: numpy.random.RandomState object
        """
        #simulate n times
        for i in range(self.n):
            #get the parameters of the compound Poisson at this time step
            self.update_cp_parameters(i)
            #simulate this compound Poisson
            self.y_array[i], self.z_array[i] = simulate_cp(
                self.poisson_rate[i], self.gamma_mean[i],
                self.gamma_dispersion[i], rng)
    
    def fit(self):
        """Fit model
        
        Fit the Compound Poisson time series to the data (model fields) and y.
            The z_array is estimated using the E step. The reg parameters are
            estimated using the M step. The compound Poisson parameters updates
            between each E and M step.
        
        Returns:
            joint log likelihood at each EM step
        """
        self.e_step()
        ln_l_array = [self.joint_log_likelihood()]
        for i in range(CompoundPoissonTimeSeries._n_em):
            print("step", i)
            #do EM
            self.m_step()
            self.e_step()
            #get the log liklihood before and after the EM step
            ln_l_before = ln_l_array[i]
            ln_l_after = self.joint_log_likelihood()
            #save the log likelihood
            ln_l_array.append(ln_l_after)
            #check if the log likelihood has decreased small enough
            if ((ln_l_before - ln_l_after)/ln_l_before
                < CompoundPoissonTimeSeries._min_ln_l_ratio):
                break
        return ln_l_array
    
    def update_all_cp_parameters(self):
        """Update all the member variables of the cp variables for all time
        steps
        
        See the method update_cp_parameters(self, index)
        """
        for i in range(self.n):
            self.update_cp_parameters(i)
    
    def update_cp_parameters(self, index):
        """Update the variables of the cp variables
        
        Updates and modifies the Poisson rate, gamma mean and gama dispersion
            parameters at a given time step
        
        Args:
            index: the time step to update the parameters at
        """
        for parameter in self.cp_parameter_array.values():
            parameter.update_value_array(index)
    
    def joint_log_likelihood(self):
        """Joint log likelihood
        
        Returns the joint log likelihood of the compound Poisson time series
            assuming the latent variable z are observed (via simulation or
            estimating using the E step). Requires the method
            update_all_cp_parameters() to be called beforehand or
            update_cp_parameters(index) for index in range(self.n). Note that
            this is done after calling e_step(), thus this method can be called
            without any prerequisites afer calling e_step(). 
        
        Returns:
            log likelihood
        """
        ln_l_array = np.zeros(self.n)
        for i in range(self.n):
            ln_l_array[i] = -self.poisson_rate[i]
            z = self.z_array[i]
            if z > 0:
                y = self.y_array[i]
                gamma_mean = self.gamma_mean[i]
                gamma_dispersion = self.gamma_dispersion[i]
                terms = np.zeros(6)
                terms[0] = (-z *
                    (math.log(gamma_mean) + math.log(gamma_dispersion))
                    /gamma_dispersion)
                terms[1] = -loggamma(z/gamma_dispersion)
                terms[2] = (z/gamma_dispersion-1)*math.log(y)
                terms[3] = -y/gamma_mean/gamma_dispersion
                terms[4] = z*math.log(self.poisson_rate[i])
                terms[5] = -loggamma(z+1)
                ln_l_array[i] += np.sum(terms)
        return np.sum(ln_l_array)
    
    def e_step(self):
        """Does the E step of the EM algorithm
        
        Make estimates of the z and updates the poisson rate, gamma mean and
            gamma dispersion parameters for each time step. Modifies the member
            variables z_array, poisson_rate, gamma_mean, and gamma_dispersion
        """
        #for each data point (forwards in time)
        for i in range(self.n):
            #update the parameter at this time step
            self.update_cp_parameters(i)
            
            #get the parameters
            poisson_rate = self.poisson_rate[i]
            gamma_mean = self.gamma_mean[i]
            gamma_dispersion = self.gamma_dispersion[i]
            #save the Tweedie parameters
            p = (1+2*gamma_dispersion) / (1+gamma_dispersion)
            phi = ((gamma_dispersion+1) * math.pow(poisson_rate, 1-p)
                * math.pow(gamma_mean, 2-p))
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
    
    def m_step(self):
        """Does the M step of the EM algorithm
        
        Estimates the reg parameters given the observed rainfall y and the
            latent variable z using gradient descent. The objective function is
            the log likelihood assuming the latent variables are observed
        """
        #set their member variables containing the gradients to be zero, they
            #are calculated in the next step
        for parameter in self.cp_parameter_array.values():
            parameter.reset_d_parameter_self_array()
        #for each time step, work out d itself / d parameter where itself can be
            #poisson_rate or gamma_mean for example and parameter can be the AR
            #or MA parameters
        #gradient at time step i depends on gradient at time step i-1, therefore
            #use loop over range(n)
        for i in range(self.n):
            for parameter in self.cp_parameter_array.values():
                parameter.calculate_d_parameter_self_i(i)
        for parameter in self.cp_parameter_array.values():
            #do gradient descent
            parameter.gradient_descent(CompoundPoissonTimeSeries._step_size)
    
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
        alpha = 1/self.gamma_dispersion[index]
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
        """Works out the compound Poisson sum, only important terms are summed.
            See Dynn, Smyth (2005)
        
        Args:
            index: time step, y[index] must be positive
            z_pow: 0,1,2,..., used for taking the sum for z^z_pow * Wy which is
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
        #the first term is 1, that is exp[ln(w_z_max)-ln(w_z_max)] = 1;
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
        
        #calculate the compound poisson terms starting at z_l and working
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
    
    class CompoundPoissonParameter:
        """Dynamic parameters of the compound Poisson
        
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
            value_array: array of values of the parameter for each time step
            _parent: CompoundPoissonTimeSeries object containing this
            _d_parameter_self_array: dictionary containing arrays of derivates
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
            self.value_array = None
            self._parent = None
            self._d_reg_self_array = None
        
        def assign_parent(self, parent):
            """Assign parent
            
            Assign the member variable _parent which points to the
                CompoundPoissonTimeSeries object which owns self
            """
            self._parent = parent
            self.n = parent.n
            self.value_array = np.zeros(self.n)
        
        def copy(self):
            """Return deep copy of itself
            """
            copy = Parameters(self.n_dim)
            for key, value in self.reg_parameters.items():
                if type(value) is np.ndarray:
                    copy[key] = value.copy()
                else:
                    copy[key] = value
            copy.value_array = self.value_array.copy()
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
            exponent += np.dot(self["reg"], self._parent.x[index,:])
            if "AR" in self.reg_parameters.keys():
                exponent += self["AR"] * self.ar_term(index)
            if "MA" in self.reg_parameters.keys():
                exponent += self["MA"] * self.ma_term(index)
            #exp link function, make it positive
            self[index] = math.exp(exponent)
        
        def ar_term(self, index):
            """AR term at a time step
            
            Returns the autoregressive term, log(self[index-1]) - constant term
            
            Args:
                index: time step
            
            Returns:
                the AR term at a time step
            """
            if index > 0:
                return math.log(self[index-1]) - self["const"]
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
        
        def gradient_descent(self, step_size):
            """Update itself using gradient gradient_descent
            
            Modifies self.reg_parameters using gradient_descent given the model
                fields and z_array
            
            Args:
                step_size: the size of the gradient to update the reg parameters
            """
            gradient = self.d_reg_ln_l()
            for key in self.reg_parameters.keys():
                self[key] += (step_size/self.n)*gradient[key]
        
        def d_reg_ln_l(self):
            """Returns the derivate of the log likelihood wrt reg parameters
            
            Returns a dictionary of the derivate of the log likelihood wrt reg
                parameters. Requires the following methods to be called
                beforehand:
                -reset_d_parameter_self_array()
                -calculate_d_parameter_self_i(index) where the outer loop is
                    for each time step and the inner loop is for each reg
                    parameter. This is required because the derivate of
                    GammaMean.ma_term[index] depends on
                    GammaDispersion._d_reg_self_array[index-1]. As a result, the
                    derivate of all reg parameters must be found together for
                    each time step
                See the method CompoundPoissonTimeSeries.m_step(self) to see
                example on how to call this method
            
            Returns:
                dictionary of the derivate of the log likelihood wrt reg
                    parameters with keys "reg", "const" as well as if available
                    "AR" and "MA"
            """
            d_self_ln_l = self.d_self_ln_l()
            d_reg_ln_l = self._d_reg_self_array
            for key in self._d_reg_self_array.keys():
                for i in range(self.n):
                    #chain rule
                    d_reg_ln_l[key][i] = (
                        self._d_reg_self_array[key][i] * d_self_ln_l[i])
                #sum over all time steps (chain rule for partial diff)
                value = d_reg_ln_l[key]
                if value.ndim == 1:
                    self._d_reg_self_array[key] = np.sum(value)
                else:
                    self._d_reg_self_array[key] = np.sum(value, 0)
            #put the gradient in a CompoundPoissonParameter object and return it
            gradient = (
                CompoundPoissonTimeSeries.CompoundPoissonParameter(self.n_dim))
            gradient.reg_parameters = self._d_reg_self_array
            return gradient
        
        def d_self_ln_l(self):
            """Derivate of the log likelihood wrt itself for each time step
            
            TO BE IMPLEMENTED
            
            Returns:
                vector of gradients
            """
            pass
        
        def reset_d_parameter_self_array(self):
            """Reset _d_parameter_self_array
            
            Set all values in _d_parameter_self_array to be numpy zeros
            """
            self._d_reg_self_array = {}
            for key, value in self.reg_parameters.items():
                self._d_reg_self_array[key] = np.zeros(
                    (self.n, value.size))
        
        def calculate_d_parameter_self_i(self, index):
            """Calculates the derivate of itself wrt parameter
            
            Modifies the member variable _d_parameter_self_array with the
                gradient of itself wrt a reg parameter. Before an iteration,
                call the method reset_d_parameter_self_array(self) to set
                everything in _d_parameter_self_array to be zeros
            
            Args:
                index: time step to modify the array _d_parameter_self_array
            """
            parameter_i = self[index]
            keys = self._d_reg_self_array.keys()
            for key in self.reg_parameters.keys():
                d_parameter_self = self._d_reg_self_array[key]
                if index > 0:
                    #AR and MA terms
                    parameter_i_1 = self[index-1]
                    if "AR" in keys:
                        d_parameter_self[index] += (self["AR"]
                            * d_parameter_self[index-1] / parameter_i_1)
                    if "MA" in keys:
                        d_parameter_self[index] += (self["MA"]
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
                        d_parameter_self[index] -= self["AR"]
                d_parameter_self[index] *= parameter_i
        
        def d_parameter_ma(self, index, key):
            """Derivate of the MA term at a time step
            
            TO BE IMPLEMENTED
            
            Args:
                index: time step
                key: name of the parameter to derivate wrt
            
            Returns:
                derivate of the MA term
            
            """
            pass
        
        def __str__(self):
            #print reg_parameters
            return self.reg_parameters.__str__()
        
        def __getitem__(self, key):
            #can return a value in value_array when key is an integer
            #can return a reg parameter when provided with "reg", "AR", "MA",
                #"const"
            if key in self.reg_parameters.keys():
                return self.reg_parameters[key]
            else:
                return self.value_array[key]
        
        def __setitem__(self, key, value):
            #can set a value in value_array when key is an integer
            #can set a reg parameter when provided with "reg", "AR", "MA",
                #"const"
            if key in self.reg_parameters.keys():
                self.reg_parameters[key] = value
            else:
                self.value_array[key] = value
    
    class PoissonRate(CompoundPoissonParameter):
        def __init__(self, n_dim):
            super().__init__(n_dim)
            self.reg_parameters["AR"] = 0.0
            self.reg_parameters["MA"] = 0.0
        def ma_term(self, index):
            if index > 0:
                poisson_rate_before = self[index-1]
                return (
                    (self._parent.z_array[index-1] - poisson_rate_before)
                    / math.sqrt(poisson_rate_before))
            else:
                return 0.0
        def d_self_ln_l(self):
            z = self._parent.z_array
            poisson_rate = self.value_array
            return z/poisson_rate - 1
        def d_parameter_ma(self, index, key):
            poisson_rate_before = self[index-1]
            return (-0.5*math.sqrt(poisson_rate_before)
                *(1+self._parent.z_array[index-1]/poisson_rate_before)
                *self._d_reg_self_array[key][index-1])
    
    class GammaMean(CompoundPoissonParameter):
        def __init__(self, n_dim):
            super().__init__(n_dim)
            self.reg_parameters["AR"] = 0.0
            self.reg_parameters["MA"] = 0.0
        def ma_term(self, index):
            if index > 0:
                z_before = self._parent.z_array[index-1]
                if z_before > 0:
                    y_before = self._parent.y_array[index-1]
                    self_before = self[index-1]
                    return (
                        (y_before/z_before - self_before)
                        / self_before
                        / math.sqrt(
                            self._parent.gamma_dispersion[index-1]
                            * z_before))
                else:
                    return 0.0
            else:
                return 0.0
        def d_self_ln_l(self):
            y = self._parent.y_array
            z = self._parent.z_array
            mu = self.value_array
            phi = self._parent.gamma_dispersion.value_array
            return (y-z*mu) / (phi*mu*mu)
        def d_parameter_ma(self, index, key):
            y = self._parent.y_array[index-1]
            z = self._parent.z_array[index-1]
            if z > 0:
                gamma_mean = self[index-1]
                gamma_dispersion = self._parent.gamma_dispersion
                d_reg_gamma_mean = (
                    self._d_reg_self_array[key][index-1])
                if key in gamma_dispersion.reg_parameters.keys():
                    d_reg_gamma_dispersion = (gamma_dispersion
                        ._d_reg_self_array[key][index-1])
                else:
                    d_reg_gamma_dispersion = 0.0
                gamma_dispersion = gamma_dispersion[index-1]
                return ( ( -(y * d_reg_gamma_mean)
                        - 0.5*(y-z*gamma_mean)*gamma_mean
                        *d_reg_gamma_dispersion
                        / gamma_dispersion
                    )
                    / ( math.pow(gamma_mean,2) * math.pow(z,3/2)
                        * math.sqrt(gamma_dispersion)))
            else:
                return 0.0
    
    class GammaDispersion(CompoundPoissonParameter):
        def __init__(self, n_dim):
            super().__init__(n_dim)
        def d_self_ln_l(self):
            d_self_ln_l = np.zeros(self.n)
            for i in range(self.n):
                z = self._parent.z_array[i]
                if z == 0:
                    d_self_ln_l[i] = 0
                else:
                    y = self._parent.y_array[i]
                    mu = self[i]
                    phi = self._parent.gamma_dispersion[i]
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
    n_dim = 2
    x = np.zeros((n,n_dim))
    for i in range(n):
        for i_dim in range(n_dim):
            x[i, i_dim] = (0.8*math.sin(2*math.pi/365 * i)
                + (i_dim+1)*rng.normal())
    
    poisson_rate = CompoundPoissonTimeSeries.PoissonRate(n_dim)
    poisson_rate["reg"] = [0.098, 0.001]
    poisson_rate["AR"] = 0.713
    poisson_rate["MA"] = 0.11
    poisson_rate["const"] = 0.355
    gamma_mean = CompoundPoissonTimeSeries.GammaMean(n_dim)
    gamma_mean["reg"] = [0.066, 0.002]
    gamma_mean["AR"] = 0.4
    gamma_mean["MA"] = 0.24
    gamma_mean["const"] = 1.3
    gamma_dispersion = CompoundPoissonTimeSeries.GammaDispersion(n_dim)
    gamma_dispersion["reg"] = [0.07, 0.007]
    gamma_dispersion["const"] = 0.373
    
    compound_poisson_time_series = CompoundPoissonTimeSeries(
        x, poisson_rate, gamma_mean, gamma_dispersion)
    compound_poisson_time_series.simulate(rng)
    
    poisson_rate_guess = math.log(
        n/(n- np.count_nonzero(compound_poisson_time_series.z_array)))
    gamma_mean_guess = (np.mean(compound_poisson_time_series.y_array)
        / poisson_rate_guess)
    gamma_dispersion_guess = (
        np.var(compound_poisson_time_series.y_array, ddof=1)
        /poisson_rate_guess/math.pow(gamma_mean_guess,2)-1)
    
    poisson_rate = CompoundPoissonTimeSeries.PoissonRate(n_dim)
    gamma_mean = CompoundPoissonTimeSeries.GammaMean(n_dim)
    gamma_dispersion = CompoundPoissonTimeSeries.GammaDispersion(n_dim)
    poisson_rate["const"] = math.log(poisson_rate_guess)
    gamma_mean["const"] = math.log(gamma_mean_guess)
    gamma_dispersion["const"] = math.log(gamma_dispersion_guess)
    
    compound_poisson_time_series.set_parameters(
        poisson_rate, gamma_mean, gamma_dispersion)
    print(compound_poisson_time_series.poisson_rate)
    print(compound_poisson_time_series.gamma_mean)
    print(compound_poisson_time_series.gamma_dispersion)
    ln_l_array = compound_poisson_time_series.fit()
    print(compound_poisson_time_series.poisson_rate)
    print(compound_poisson_time_series.gamma_mean)
    print(compound_poisson_time_series.gamma_dispersion)
    
    compound_poisson_time_series.simulate(rng)
    
    y = compound_poisson_time_series.y_array
    poisson_rate_array = compound_poisson_time_series.poisson_rate.value_array
    gamma_mean_array = compound_poisson_time_series.gamma_mean.value_array
    gamma_dispersion_array = (
        compound_poisson_time_series.gamma_dispersion.value_array)
    
    acf = stats.acf(y, nlags=100, fft=True)
    pacf = stats.pacf(y, nlags=10)
    
    plt.figure()
    plt.plot(ln_l_array)
    plt.xlabel("Number of EM steps")
    plt.ylabel("log-likelihood")
    plt.savefig("../figures/simulation_ln_l.png")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(y)
    plt.xlabel("Time (day)")
    plt.ylabel("Rainfall (mm)")
    plt.savefig("../figures/simulation_rain.png")
    plt.show()
    plt.close()
    
    for i_dim in range(n_dim):
        plt.figure()
        plt.plot(x[:,i_dim])
        plt.xlabel("Time (day)")
        plt.ylabel("Model field "+str(i_dim))
        plt.savefig("../figures/simulation_model_field_"+str(i_dim)+".png")
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
