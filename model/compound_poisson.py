import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import statsmodels.tsa.stattools as stats

from scipy.special import loggamma, digamma

class CompoundPoissonTimeSeries:
    """Compound Poisson Time Series with ARMA behaviour
    
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
        self.poisson_rate_parameter = poisson_rate_parameter
        self.gamma_mean_parameter = gamma_mean_parameter
        self.gamma_dispersion_parameter = gamma_dispersion_parameter
        self.poisson_rate_array = np.zeros(self.n)
        self.gamma_mean_array = np.zeros(self.n)
        self.gamma_dispersion_array = np.zeros(self.n)
        self.z_array = np.zeros(self.n)
        self.y_array = np.zeros(self.n)
        self._tweedie_p_array = np.zeros(self.n)
        self._tweedie_phi_array = np.zeros(self.n)
        #assign the parameters parents as self, this is so that the parameter
            #objects has access to the data 
        self.poisson_rate_parameter.assign_parent(self)
        self.gamma_mean_parameter.assign_parent(self)
        self.gamma_dispersion_parameter.assign_parent(self)
        #perpare the parameters by converting all to numpy array
        self.poisson_rate_parameter.convert_all_to_np()
        self.gamma_mean_parameter.convert_all_to_np()
        self.gamma_dispersion_parameter.convert_all_to_np()
    
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
            z_max: positive integer, index of the biggest term in the compound
                Poisson sum
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
            ln_wz: log compopund Poisson term
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
            ln_sum_w: log compound Poisson sum
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
    
    class Parameters:
        """Contains regressive, autoregressive, moving average and constant
            parameters for the compound Poisson time series model
        
        Attributes:
            parameters: dictionary with keys "reg", "AR", "MA", "const" with
                corresponding values regression parameters
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
        
        def convert_all_to_np(self):
            for key, value in self.parameters.items():
                self.parameters[key] = np.asarray(value)
        
        def d_self_ln_l(self):
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
            self._parent = parent
        
        def ar_term(self, index):
            if index > 0:
                return (math.log(self._self_array[index-1])
                    - self.parameters["const"])
            else:
                return 0.0
        
        def ma_term(self, index):
            pass
        
        def d_parameter_ma(self, index, d_parameter_self):
            pass
        
        def d_parameter_ln_l(self):
            d_self_ln_l = self.d_self_ln_l()
            d_parameter_self_array = {}
            keys = self.parameters.keys()
            for key, value in self.parameters.items():
                d_parameter_self_array[key] = np.zeros(
                    (self._parent.n, value.size))
            for i in range(self._parent.n):
                parameter_i = self._self_array[i]
                if i != 0:
                    parameter_i_1 = self._self_array[i-1]
                for key in self.parameters.keys():
                    d_parameter_self = d_parameter_self_array[key]
                    if i == 0:
                        if key == "reg":
                            d_parameter_self[i,:] = (
                                parameter_i * self._parent.x[i,:])
                        elif key == "const":
                            if "AR" in keys:
                                d_parameter_self[i] = (
                                    parameter_i * (1 - self.parameters["AR"]))
                            else:
                                d_parameter_self[i] = parameter_i
                        else:
                            d_parameter_self[i] = 0
                    elif i == 1 and key == "AR":
                        d_parameter_self[i] = (parameter_i
                            * (math.log(parameter_i_1)
                                - self.parameters["const"]))
                    elif i == 1 and key == "MA":
                        d_parameter_self[i] = (parameter_i * self.ma_term(i))
                    else:
                        if "AR" in keys:
                            d_parameter_self[i] += (
                                self.parameters["AR"] * d_parameter_self[i-1]
                                / parameter_i_1)
                        if "MA" in keys:
                            d_parameter_self[i] += (
                                self.parameters["MA"]
                                * self.d_parameter_ma(
                                    i, key, d_parameter_self_array))
                        if key == "reg":
                            d_parameter_self[i] +=  self._parent.x[i,:]
                        elif key == "AR":
                            d_parameter_self[i] += (math.log(parameter_i_1)
                                - self.parameters["const"])
                        elif key == "MA":
                            d_parameter_self[i] += self.ma_term(i)
                        elif key == "const":
                            if "AR" in keys:
                                d_parameter_self[i] += 1 - self.parameters["AR"]
                            else:
                                d_parameter_self[i] += 1
                        d_parameter_self[i] *= parameter_i
                for key in self.parameters.keys():
                    d_parameter_self_array[key][i] *= d_self_ln_l[i]
            for key in self.parameters.keys():
                if value.ndim == 1:
                    d_parameter_self_array[key] = np.sum(
                        d_parameter_self_array[key])
                else:
                    d_parameter_self_array[key] = np.sum(
                        d_parameter_self_array[key], 0)
            return d_parameter_self_array
        
        def __getitem__(self, key):
            return self.parameters[key]
        
        def __setitem__(self, key, value):
            self.parameters[key] = value

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
        def d_parameter_ma(self, index, key, d_parameter_self_array):
            poisson_rate_before = self._parent.poisson_rate_array[index-1]
            return (-0.5*math.sqrt(poisson_rate_before)
                *(1+self._parent.z_array[index-1]/poisson_rate_before)
                *d_parameter_self_array[key][index-1])

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
        def d_parameter_ma(self, index, key, d_parameter_self_array):
            y = self._parent.y_array[index-1]
            z = self._parent.z_array[index-1]
            gamma_mean = self._self_array[index-1]
            gamma_dispersion = self._parent.gamma_dispersion_array[index-1]
            return (-(y * d_parameter_self_array[""]))

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
                        +digamma(z/phi)) / math.pow(phi,2)
            return d_self_ln_l

def simulate_cp(poisson_rate, gamma_mean, gamma_dispersion, rng):
    """Simulate a single compound poisson random variable
    
    Args:
        poisson_rate: scalar
        gamma_mean: scalar
        gamma_dispersion: scalar
        rng: numpy.random.RandomState object
    
    Returns:
        y: compound Poisson random variable
        z: latent Poisson random variable
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
    
    compound_poisson_time_series.e_step()
    
    plt.figure()
    plt.plot(poisson_rate_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Poisson rate")
    plt.savefig("../figures/simulation_lambda_estimate.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(gamma_mean_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Mean of gamma")
    plt.savefig("../figures/simulation_mu_estimate.png")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(range(n),compound_poisson_time_series.z_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Z")
    plt.savefig("../figures/simulation_z_estimate.png")
    plt.show()
    plt.close()

main()
