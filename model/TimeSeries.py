import math
import numpy as np
from scipy.special import loggamma, polygamma

class TimeSeries:
    """Compound Poisson Time Series with ARMA behaviour
    
    A time series distribued as compound Poisson with dynamic varying
        parameters. Objects from this class can simulate the model and fit the
        model.
    Initalise the model by passing initial parameters via the constructor. This
        is used to simulate.
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
        z_var_array: variance of the z at each time step
        y_array: compound poisson variables at each time step
    """
    
    def __init__(self, x, cp_parameter_array):
        """
        Args:
            x: design matrix of the model fields, shape (n, n_dim)
            cp_parameter_array: array containing in order PoissonRate object,
                GammaMean object, GammaDispersion object
        """
        self.x = x
        self.n = x.shape[0]
        self.n_dim = x.shape[1]
        self.poisson_rate = None
        self.gamma_mean = None
        self.gamma_dispersion = None
        self.n_parameter = None
        #array containing poisson_rate, gamma_mean and gamma_dispersion
        self.cp_parameter_array = None
        self.z_array = np.zeros(self.n)
        self.z_var_array = np.zeros(self.n)
        self.y_array = np.zeros(self.n)
        
        self.set_new_parameter(cp_parameter_array)
    
    def set_new_parameter(self, cp_parameter_array):
        """Set the member variables of newly instantised cp paramters
        
        Set the member variables of poisson_rate, gamma_mean,
            gamma_dispersion and cp_parameter_array. Itself is assigned as the
            parent to these objects. The reg parameters in these objects are
            converted to numpy array
        """
        self.set_parameter(cp_parameter_array)
        for parameter in self.cp_parameter_array:
            #assign the parameters parents as self, this is so that the
                #parameter objects has access to all member variables and
                #methods
            parameter.assign_parent(self)
            #prepare the parameters by converting all to numpy array
            parameter.convert_all_to_np()
        #get number of parameters in this model
        self.n_parameter = 0
        for parameter in self.cp_parameter_array:
            for reg in parameter.values():
                self.n_parameter += reg.shape[0]
    
    def set_parameter(self, cp_parameter_array):
        """Set the member variables of the cp paramters
        
        Set the member variables of poisson_rate, gamma_mean,
            gamma_dispersion and cp_parameter_array. 
        
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
        """
        cp_parameter_array = []
        for parameter in self.cp_parameter_array:
            cp_parameter_array.append(parameter.copy())
        return cp_parameter_array
    
    def get_parameter_vector(self, cp_parameter_array=None):
        """Return the regression parameters as one vector
        """
        if cp_parameter_array is None:
            cp_parameter_array = self.cp_parameter_array
        #for each parameter, concatenate vectors together
        parameter_vector = np.zeros(0)
        for parameter in cp_parameter_array:
            parameter_vector = np.concatenate(
                (parameter_vector, parameter.get_reg_vector()))
        return parameter_vector
    
    def set_parameter_vector(self, parameter_vector):
        """Set the regression parameters using a single vector
        """
        parameter_counter = 0 #count the number of parameters
        #for each parameter, take the correction section of parameter_vector and
            #put it in the parameter
        for parameter in self.cp_parameter_array:
            n_parameter = 0
            for reg in parameter.values():
                n_parameter += reg.shape[0]
            parameter.set_reg_vector(
                parameter_vector[
                parameter_counter:parameter_counter+n_parameter])
            parameter_counter += n_parameter
    
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
        pass
    
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
    
    def get_joint_log_likelihood(self):
        """Joint log likelihood
        
        Returns the joint log likelihood of the compound Poisson time series
            assuming the latent variable z are observed (via simulation or
            estimating using the E step). Requires the method
            update_all_cp_parameters() to be called beforehand or
            update_cp_parameters(index) for index in range(self.n) if only a few
            parameters has changed. Note that this is done after calling
            e_step(), thus this method can be called without any prerequisites
            afer calling e_step(). 
        
        Returns:
            log likelihood
        """
        ln_l_array = np.zeros(self.n)
        for i in range(self.n):
            ln_l_array[i] = self.get_joint_log_likelihood_i(i)
        return np.sum(ln_l_array)
    
    def get_em_objective(self):
        """Return M step objective for a single data point
        
        Requires the method update_all_cp_parameters() to be called beforehand
            or update_cp_parameters(index) for index in range(self.n).
        """
        ln_l_array = np.zeros(self.n)
        for i in range(self.n):
            ln_l_array[i] = self.get_em_objective_i(i)
        return np.sum(ln_l_array)
    
    def get_em_objective_i(self, i):
        """Return M step objective for a single data point
        
        Requires the method update_all_cp_parameters() to be called beforehand
            or update_cp_parameters(index) for index in range(self.n).
        """
        objective = self.get_joint_log_likelihood_i(i)
        z = self.z_array[i]
        if z > 0:
            z_var = self.z_var_array[i]
            gamma_dispersion = self.gamma_dispersion[i]
            objective -= (0.5*z_var*polygamma(1, z/gamma_dispersion)
                /math.pow(gamma_dispersion, 2))
        return objective
    
    def get_joint_log_likelihood_i(self, i):
        """Return joint log likelihood for a single data point
        
        Requires the method update_all_cp_parameters() to be called beforehand
            or update_cp_parameters(index) for index in range(self.n).
        """
        ln_l = -self.poisson_rate[i]
        z = self.z_array[i]
        if z > 0:
            y = self.y_array[i]
            gamma_mean = self.gamma_mean[i]
            gamma_dispersion = self.gamma_dispersion[i]
            cp_term = TimeSeries.Terms(self, i)
            terms = np.zeros(3)
            terms[0] = -y/gamma_mean/gamma_dispersion
            terms[1] = -math.log(y)
            terms[2] = cp_term.ln_wz(z)
            ln_l += np.sum(terms)
        return ln_l
    
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
                sum = TimeSeries.Terms(self, i)
                normalisation_constant = sum.ln_sum_w()
                #work out the expectation
                sum = TimeSeries.TermsZ(self, i)
                self.z_array[i] = math.exp(
                    sum.ln_sum_w() - normalisation_constant)
                sum = TimeSeries.TermsZ2(self, i)
                self.z_var_array[i] = (math.exp(
                    sum.ln_sum_w() - normalisation_constant)
                    - math.pow(self.z_array[i],2))
        
    class Terms:
        """Contains the terms for the compound Poisson series.
        
        Can sum the compound Poisson series by summing only important terms.
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
                    if log_ratio > TimeSeries.Terms.cp_sum_threshold:
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
                if log_ratio > TimeSeries.Terms.cp_sum_threshold:
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
    
    class TermsZ(Terms):
        def __init__(self, parent, index):
            super().__init__(parent, index)
        def log_expectation_term(self, z):
            return math.log(z)
    
    class TermsZ2(Terms):
        def __init__(self, parent, index):
            super().__init__(parent, index)
        def log_expectation_term(self, z):
            return 2*math.log(z)

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
