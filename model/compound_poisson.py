import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import statsmodels.tsa.stattools as stats

from scipy.special import loggamma, digamma, polygamma

class TimeSeries:
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
        cp_parameter_array: array pointing to poisson_rate, gamma_mean and
            gamma_dispersion
        z_array: latent poisson variables at each time step
        y_array: compound poisson variables at each time step
        ln_l_array: the joint log likelihood after calling the method fit()
        step_size: the step size for gradient descent, used in the M step
        n_em: number of EM steps
        n_gradient_descent: number of steps in a M step
        min_ln_l_ratio: determines when to stop the EM algorithm if the log
            likelihood increases not very much
    """
    
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
        #array containing poisson_rate, gamma_mean and gamma_dispersion
        self.cp_parameter_array = None
        self.z_array = np.zeros(self.n)
        self.z_var_array = np.zeros(self.n)
        self.y_array = np.zeros(self.n)
        self.ln_l_array = None
        
        self.step_size = 0.1
        self.n_em = 100
        self.n_gradient_descent = 100
        self.min_ln_l_ratio = 0.0001
        
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
        self.cp_parameter_array = [poisson_rate, gamma_mean, gamma_dispersion]
        
        for parameter in self.cp_parameter_array:
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
            between each E and M step. The joint log likelihood at each EM step
            can be obtained from the member variable ln_l_array
        """
        self.e_step()
        self.ln_l_array = [self.joint_log_likelihood()]
        for i in range(self.n_em):
            print("step", i)
            #do EM
            self.m_step(self.ln_l_array[len(self.ln_l_array)-1])
            self.e_step()
            #save the log likelihood
            self.ln_l_array.append(self.joint_log_likelihood())
            #check if the log likelihood has decreased small enough
            if self.has_converge(self.ln_l_array):
                break
    
    def has_converge(self, ln_l_array):
        """Set convergence criterion from the likelihood
        
        During the EM algorithm, the algorithm may stop if the log likelihood
            does not increase any more. This function returns True if this is
            the case
        
        Args:
            ln_l_array: array of log likelihoods at each step, does not include
                future steps
        
        Returns:
            boolean, True if the likelihood is considered converged
        """
        ln_l_before = ln_l_array[len(ln_l_array)-2]
        ln_l_after = ln_l_array[len(ln_l_array)-1]
        return (ln_l_before - ln_l_after)/ln_l_before < self.min_ln_l_ratio
    
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
        for parameter in self.cp_parameter_array:
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
                z_var = self.z_var_array[i]
                poisson_rate = self.poisson_rate[i]
                gamma_mean = self.gamma_mean[i]
                gamma_dispersion = self.gamma_dispersion[i]
                terms = np.zeros(7)
                terms[0] = (-z/gamma_dispersion *
                    (math.log(gamma_mean) + math.log(gamma_dispersion)))
                terms[1] = -loggamma(z/gamma_dispersion)
                terms[2] = (z/gamma_dispersion-1)*math.log(y)
                terms[3] = -y/gamma_mean/gamma_dispersion
                terms[4] = z*math.log(poisson_rate)
                terms[5] = -loggamma(z+1)
                terms[6] = -(0.5*z_var*polygamma(1, z/gamma_dispersion)
                    /math.pow(gamma_dispersion, 2))
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
            
            #if the rainfall is zero, then z is zero (it has not rained)
            if self.y_array[i] == 0:
                self.z_array[i] = 0
                self.z_var_array[i] = 0
            else:
                #work out the normalisation constant for the expectation
                sum = TimeSeries.Sum(self, i)
                normalisation_constant = sum.ln_sum_w()
                #work out the expectation
                sum = TimeSeries.SumZ(self, i)
                self.z_array[i] = math.exp(
                    sum.ln_sum_w() - normalisation_constant)
                sum = TimeSeries.SumZ2(self, i)
                self.z_var_array[i] = (math.exp(
                    sum.ln_sum_w() - normalisation_constant)
                    - math.pow(self.z_array[i],2))
    
    def m_step(self, ln_l):
        """Does the M step of the EM algorithm
        
        Estimates the reg parameters given the observed rainfall y and the
            latent variable z using gradient descent. The objective function is
            the log likelihood assuming the latent variables are observed
        """
        ln_l_array = [ln_l]
        #do gradient descent multiple times, keep track of log likelihood
        for i in range(self.n_gradient_descent):
            #set the member variables containing the gradients to be zero,
                #they are calculated in the next step
            for parameter in self.cp_parameter_array:
                parameter.reset_d_reg_self_array()
            #for each time step, work out d itself / d parameter where itself
                #can be poisson_rate or gamma_mean for example and parameter can
                #be the AR or MA parameters
            #gradient at time step i depends on gradient at time step i-1,
                #therefore work out each gradient in sequence
            for i in range(self.n):
                for parameter in self.cp_parameter_array:
                    parameter.calculate_d_reg_self_i(i)
            for parameter in self.cp_parameter_array:
                #do gradient descent
                parameter.gradient_descent()
            #work out log likelihood and test for convergence
            self.update_all_cp_parameters()
            ln_l_array.append(self.joint_log_likelihood())
            if self.has_converge(ln_l_array):
                break
    
    class Sum:
        
        """Works out the compound Poisson sum, only important terms are summed.
            See Dynn, Smyth (2005)
        
        Attributes:
            y
            poisson_rate
            gamma_mean
            gamma_dispersion
        """
        #negative number, determines the smallest term to add in the compound
            #Poisson sum
        cp_sum_threshold = -37
        
        def __init__(self, parent, index):
            self.y = parent.y_array[index]
            self.poisson_rate = parent.poisson_rate[index]
            self.gamma_mean = parent.gamma_mean[index]
            self.gamma_dispersion = parent.gamma_dispersion[index]
        
        def log_expectation_term(self, z):
            """Multiple each term in the sum by exp(this return value)
            
            Can be override if want to take for example expectation
            """
            return 0
        
        def ln_sum_w(self):
            """Works out the compound Poisson sum, only important terms are
                summed. See Dynn, Smyth (2005).
                
            Returns:
                log compound Poisson sum
            """
            
            #get the y with the biggest term in the compound Poisson sum
            z_max = self.z_max()
            #get the biggest log compound Poisson term + any expectation terms
            ln_w_max = self.ln_wz(z_max) + self.log_expectation_term(z_max)
            
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
                #if the lower bound is 0, then set is_got_z_l to be true and
                    #raise the lower bound back by one
                if z_l == 0:
                    is_got_z_l = True
                    z_l += 1
                else: #else the lower bound is not 0
                    #calculate the log ratio of the compound poisson term with
                        #the maximum compound poisson term
                    log_ratio = np.sum(
                        [self.ln_wz(z_l), 
                        self.log_expectation_term(z_l), 
                        -ln_w_max])
                    #if this log ratio is bigger than the threshold
                    if log_ratio > TimeSeries.Sum.cp_sum_threshold:
                        #append the ratio to the array of terms
                        terms.append(math.exp(log_ratio))
                    else:
                        #else the log ratio is smaller than the threshold
                        #set is_got_z_l to be true and raise the lower bound by
                            #1
                        is_got_z_l = True
                        z_l += 1
            
            #while we haven't got an upper bound
            while not is_got_z_u:
                #raise the upper bound by 1
                z_u += 1;
                #calculate the log ratio of the compound poisson term with the
                    #maximum compound poisson term
                log_ratio = np.sum(
                    [self.ln_wz(z_u),
                    self.log_expectation_term(z_u),
                    -ln_w_max])
                #if this log ratio is bigger than the threshold
                if log_ratio > TimeSeries.Sum.cp_sum_threshold:
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
        
        def z_max(self):
            """Gets the index of the biggest term in the compound Poisson sum
            
            Returns:
                positive integer, index of the biggest term in the compound
                    Poisson sum
            """
            #get the optima with respect to the sum index, then round it to get
                #an integer
            terms = np.zeros(3)
            terms[0] = math.log(self.y)
            terms[1] = self.gamma_dispersion * math.log(self.poisson_rate)
            terms[2] = -math.log(self.gamma_mean)
            z_max = math.exp(np.sum(terms)/(self.gamma_dispersion+1))
            z_max = round(z_max)
            #if the integer is 0, then set the index to 1
            if z_max == 0:
                z_max = 1
            return z_max
        
        def ln_wz(self, z):
            """Return a log term from the compound Poisson sum
            
            Args:
                index: time step, y[index] must be positive
                z: Poisson variable or index of the sum element
            
            Returns:
                log compopund Poisson term
            """
            
            #declare array of terms to be summed to work out ln_wz
            terms = np.zeros(6)
            
            #work out each individual term
            terms[0] = -z*math.log(self.gamma_dispersion)/self.gamma_dispersion
            terms[1] = -z*math.log(self.gamma_mean)/self.gamma_dispersion
            terms[2] = -loggamma(z/self.gamma_dispersion)
            terms[3] = z*math.log(self.y)/self.gamma_dispersion
            terms[4] = z*math.log(self.poisson_rate)
            terms[5] = -loggamma(1+z)
            #sum the terms to get the log compound Poisson sum term
            ln_wz = np.sum(terms)
            return ln_wz
    
    class SumZ(Sum):
        def __init__(self, parent, index):
            super().__init__(parent, index)
        def log_expectation_term(self, z):
            return math.log(z)
    
    class SumZ2(Sum):
        def __init__(self, parent, index):
            super().__init__(parent, index)
        def log_expectation_term(self, z):
            return 2*math.log(z)
    
    class SumZDigamma(Sum):
        def __init__(self, parent, index):
            super().__init__(parent, index)
        def log_expectation_term(self, z):
            return math.log(z) + math.log(digamma(z/self.gamma_dispersion))
    
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
        
        def copy(self):
            """Return deep copy of itself
            """
            copy = self.__class__(self.n_dim)
            for key, value in self.reg_parameters.items():
                if type(value) is np.ndarray:
                    copy[key] = value.copy()
                else:
                    copy[key] = value
            copy.assign_parent(self._parent)
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
        
        def gradient_descent(self):
            """Update itself using gradient descent
            
            Modifies self.reg_parameters using gradient_descent given the model
                fields and z_array. Requires calculate_d_reg_self_i() to be
                called for all index. See the method m_step in TimeSeries
            """
            gradient = self.d_reg_ln_l()
            for key in self.reg_parameters.keys():
                self[key] += (self._parent.step_size/self.n)*gradient[key]
        
        def stochastic_gradient_descent(self, index):
            """Update itself using stochastic gradient descent
            
            Modifies self.reg_parameters using gradient_descent given the model
                fields and z_array. Requires calculate_d_reg_self_i() to be
                called for all index up to the args index. See the method m_step
                in TimeSeriesSgd
            
            Args:
                index: the point in the time series to be used for stochastic
                    gradient descent
            """
            d_self_ln_l = self.d_self_ln_l(index)
            for key in self._d_reg_self_array.keys():
                self[key] += (self._parent.stochastic_step_size
                    * (self._d_reg_self_array[key][index]* d_self_ln_l))
        
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
            for key in self._d_reg_self_array.keys():
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
            gradient = TimeSeries.Parameter(self.n_dim)
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
            keys = self._d_reg_self_array.keys()
            for key in self.reg_parameters.keys():
                d_reg_self = self._d_reg_self_array[key]
                if index > 0:
                    #AR and MA terms
                    parameter_i_1 = self[index-1]
                    if "AR" in keys:
                        d_reg_self[index] += (self["AR"]
                            * d_reg_self[index-1] / parameter_i_1)
                    if "MA" in keys:
                        d_reg_self[index] += (self["MA"]
                            * self.d_parameter_ma(index, key))
                if key == "reg":
                    d_reg_self[index] += self._parent.x[index,:]
                elif key == "AR":
                    d_reg_self[index] += self.ar_term(index)
                elif key == "MA":
                    d_reg_self[index] += self.ma_term(index)
                elif key == "const":
                    d_reg_self[index] += 1
                    if "AR" in keys and index > 0:
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
    
    class PoissonRate(Parameter):
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
        def ma_term(self, index):
            if index > 0:
                z_before = self._parent.z_array[index-1]
                if z_before > 0:
                    y_before = self._parent.y_array[index-1]
                    self_before = self[index-1]
                    return (
                        (y_before - z_before*self_before)
                        / self_before
                        / math.sqrt(
                            self._parent.gamma_dispersion[index-1]
                            * z_before))
                else:
                    return 0.0
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
                if key in gamma_dispersion.reg_parameters.keys():
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

class TimeSeriesSgd(TimeSeries):
    """Compound Poisson Time Series which uses stochastic gradient descent
    
    The fitting is done using different initial values. Regular gradient descent
        is used on the first value in the EM algorithm. Different initial values
        are obtained by using stochastic gradient descent in the M step.
    
    Attributes:
        n_initial: number of different inital points to try out
        stochastic_step_size: step size of stochastic gradient descent
        n_stochastic_step: number of stochastic gradient descent steps
        ln_l_max: points to the selected maximum log likelihood from the member
            variable ln_l_array
        ln_l_stochastic_index: array which points to the member variable
            ln_l_array which indicates which are from gradient descent and
            stochastic gradient descent. [1, ..., len(ln_l_array)]. The values
            in the middle correspond to resulting first log likelihood from
            gradient descent or stochastic gradient descent
        _rng: random number generator to generate a random permutation to select
            time points at random
        _permutation_iter: iterator for the permutation of index
    """
    
    def __init__(self, x, poisson_rate, gamma_mean, gamma_dispersion):
        super().__init__(x, poisson_rate, gamma_mean, gamma_dispersion)
        self.n_initial = 100
        self.stochastic_step_size = 0.01
        self.n_stochastic_step = 10
        self.ln_l_max_index = 0
        self.ln_l_stochastic_index = [1]
        self._rng = random.RandomState(np.uint32(672819639))
        self._permutation_iter = self._rng.permutation(self.n).__iter__()
    
    def fit(self):
        """Fit model - override
        
        Regular gradient descent is used on the first value in the EM algorithm.
            Different initial values are obtained by using stochastic gradient
            descent in the M step. Select the parameters which has the maximum
            log likelihood.
        """
        ln_l_all_array = [] #array of all log likelihoods
        ln_l_max = float("-inf") #keep track of the maximum likelihood
        cp_parameter_array = None #parameters with the maximum likelihood
        #for multiple initial values
        for i in range(self.n_initial):
            print("initial value", i)
            print("gradient descent")
            super().fit() #regular gradient descent
            #copy the log likelihood
            for ln_l in self.ln_l_array:
                ln_l_all_array.append(ln_l)
            #check for convergence in the log likelihood
            ln_l = ln_l_all_array[len(ln_l_all_array)-1]
            if ln_l > ln_l_max:
                #the log likelihood is bigger, copy the parmeters
                ln_l_max = ln_l
                self.ln_l_max_index = len(ln_l_all_array)-1
                cp_parameter_array = [
                    self.poisson_rate.copy(),
                    self.gamma_mean.copy(),
                    self.gamma_dispersion.copy(),
                ]
            for parameter in cp_parameter_array:
                print(parameter)
            print("stochastic gradient descent")
            #do stochastic gradient descent to get a different initial value
            if i < self.n_initial-1:
                #track when stochastic gradient descent was done for this entry
                    #of ln_l_array
                self.ln_l_stochastic_index.append(len(ln_l_all_array))
                for j in range(self.n_stochastic_step):
                    print("step", j)
                    self.m_stochastic_step()
                    self.update_all_cp_parameters()
                    ln_l_all_array.append(self.joint_log_likelihood())
                #track when gradient descent was done
                #the E step right after this in super().fit() is considered part
                    #of stochastic gradient descent
                self.ln_l_stochastic_index.append(len(ln_l_all_array)+1)
            else:
                self.ln_l_stochastic_index.append(len(ln_l_all_array))
        #copy results to the member variable
        self.ln_l_array = ln_l_all_array
        self.cp_parameter_array = cp_parameter_array
        self.poisson_rate = cp_parameter_array[0]
        self.gamma_mean = cp_parameter_array[1]
        self.gamma_dispersion = cp_parameter_array[2]
        self.e_step()
    
    def m_stochastic_step(self):
        """Does stochastic gradient descent
        """
        #set their member variables containing the gradients to be zero, they
            #are calculated in the next step
        for parameter in self.cp_parameter_array:
            parameter.reset_d_reg_self_array()
        index = self.get_next_permutation()
        #work out the gradients for all points up to index
        for i in range(index+1):
            for parameter in self.cp_parameter_array:
                parameter.calculate_d_reg_self_i(i)
        for parameter in self.cp_parameter_array:
            #do stochastic gradient descent
            parameter.stochastic_gradient_descent(index)
    
    def get_next_permutation(self):
        """Returns a index from a random permutation
        
        Returns an interger which is from a random permutation. Once all index
            has been used, a new permutation is generated
        
        Returns:
            integer, index from random permutation
        """
        try:
            return self._permutation_iter.__next__()
        except StopIteration:
            self._permutation_iter = self._rng.permutation(self.n).__iter__()
            return self._permutation_iter.__next__()

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
    scale = gamma_mean * gamma_dispersion #gamma scale parameter
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
    
    poisson_rate = TimeSeries.PoissonRate(n_dim)
    poisson_rate["reg"] = [0.098, 0.001]
    poisson_rate["AR"] = 0.13
    poisson_rate["MA"] = 0.19
    poisson_rate["const"] = 0.42
    gamma_mean = TimeSeries.GammaMean(n_dim)
    gamma_mean["reg"] = [0.066, 0.002]
    gamma_mean["AR"] = 0.1
    gamma_mean["MA"] = 0.1
    gamma_mean["const"] = 0.89
    gamma_dispersion = TimeSeries.GammaDispersion(n_dim)
    gamma_dispersion["reg"] = [0.07, 0.007]
    gamma_dispersion["const"] = 0.12
    
    time_series = TimeSeriesSgd(x, poisson_rate, gamma_mean, gamma_dispersion)
    time_series.simulate(rng)
    print_figures(time_series, "simulation")
    
    poisson_rate_guess = math.log(n/(n- np.count_nonzero(time_series.z_array)))
    gamma_mean_guess = np.mean(time_series.y_array) / poisson_rate_guess
    gamma_dispersion_guess = (np.var(time_series.y_array, ddof=1)
        /poisson_rate_guess/math.pow(gamma_mean_guess,2)-1)
    
    poisson_rate = TimeSeries.PoissonRate(n_dim)
    gamma_mean = TimeSeries.GammaMean(n_dim)
    gamma_dispersion = TimeSeries.GammaDispersion(n_dim)
    poisson_rate["const"] = math.log(poisson_rate_guess)
    gamma_mean["const"] = math.log(gamma_mean_guess)
    gamma_dispersion["const"] = math.log(gamma_dispersion_guess)
    
    time_series.set_parameters(poisson_rate, gamma_mean, gamma_dispersion)
    print(time_series.poisson_rate)
    print(time_series.gamma_mean)
    print(time_series.gamma_dispersion)
    time_series.fit()
    
    plt.figure()
    ax = plt.gca()
    ln_l_array = time_series.ln_l_array
    ln_l_stochastic_index = time_series.ln_l_stochastic_index
    for i in range(len(ln_l_stochastic_index)-1):
        start = ln_l_stochastic_index[i]-1
        end = ln_l_stochastic_index[i+1]
        if i%2 == 0:
            linestyle = "-"
        else:
            linestyle = ":"
        ax.set_prop_cycle(None)
        plt.plot(range(start, end), ln_l_array[start:end], linestyle=linestyle)
    plt.axvline(x=time_series.ln_l_max_index, linestyle='--')
    plt.xlabel("Number of EM steps")
    plt.ylabel("log-likelihood")
    plt.savefig("../figures/fit_ln_l.pdf")
    plt.close()
    print(time_series.poisson_rate)
    print(time_series.gamma_mean)
    print(time_series.gamma_dispersion)
    
    time_series.simulate(rng)
    print_figures(time_series, "fitted")

def print_figures(time_series, prefix):
    
    x = time_series.x
    y = time_series.y_array
    z = time_series.z_array
    n = time_series.n
    n_dim = time_series.n_dim
    poisson_rate_array = time_series.poisson_rate.value_array
    gamma_mean_array = time_series.gamma_mean.value_array
    gamma_dispersion_array = time_series.gamma_dispersion.value_array
    
    acf = stats.acf(y, nlags=100, fft=True)
    pacf = stats.pacf(y, nlags=10)
    
    plt.figure()
    plt.plot(y)
    plt.xlabel("Time (day)")
    plt.ylabel("Rainfall (mm)")
    plt.savefig("../figures/"+prefix+"_rain.pdf")
    plt.close()
    
    for i_dim in range(n_dim):
        plt.figure()
        plt.plot(x[:,i_dim])
        plt.xlabel("Time (day)")
        plt.ylabel("Model field "+str(i_dim))
        plt.savefig("../figures/"+prefix+"_model_field_"+str(i_dim)+".pdf")
        plt.close()
    
    plt.figure()
    plt.bar(np.asarray(range(acf.size)), acf)
    plt.xlabel("Time (day)")
    plt.ylabel("Autocorrelation")
    plt.savefig("../figures/"+prefix+"_acf.pdf")
    plt.close()
    
    plt.figure()
    plt.bar(np.asarray(range(pacf.size)), pacf)
    plt.xlabel("Time (day)")
    plt.ylabel("Partial autocorrelation")
    plt.savefig("../figures/"+prefix+"_pacf.pdf")
    plt.close()
    
    plt.figure()
    plt.plot(poisson_rate_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Poisson rate")
    plt.savefig("../figures/"+prefix+"_lambda.pdf")
    plt.close()
    
    plt.figure()
    plt.plot(gamma_mean_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Gamma mean (mm)")
    plt.savefig("../figures/"+prefix+"_mu.pdf")
    plt.close()
    
    plt.figure()
    plt.plot(gamma_dispersion_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Gamma dispersion")
    plt.savefig("../figures/"+prefix+"_dispersion.pdf")
    plt.close()
    
    plt.figure()
    plt.plot(range(n), z)
    plt.xlabel("Time (day)")
    plt.ylabel("Z")
    plt.savefig("../figures/"+prefix+"_z.pdf")
    plt.close()
    
main()
