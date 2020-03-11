import numpy as np
from numpy import random

from compound_poisson import time_series

class TimeSeriesGd(time_series.TimeSeries):
    """Compound Poisson Time Series with ARMA behaviour
    
    Attributes:
        ln_l_array: the joint log likelihood after calling the method fit()
        step_size: the step size for gradient descent, used in the M step
        n_em: number of EM steps
        n_gradient_descent: number of steps in a M step
        min_ln_l_ratio: determines when to stop the EM algorithm if the log
            likelihood increases not very much
    """
    
    def __init__(self, 
                 x,
                 rainfall=None,
                 poisson_rate_n_arma=None,
                 gamma_mean_n_arma=None,
                 cp_parameter_array=None):
        super().__init__(x,
                         rainfall,
                         poisson_rate_n_arma,
                         gamma_mean_n_arma,
                         cp_parameter_array)
        self.ln_l_array = None
        self.step_size = 0.1
        self.n_em = 100
        self.n_gradient_descent = 100
        self.min_ln_l_ratio = 0.0001
    
    def fit(self):
        """Fit model
        
        Fit the Compound Poisson time series to the data (model fields) and y.
            The z_array is estimated using the E step. The reg parameters are
            estimated using the M step. The compound Poisson parameters updates
            between each E and M step. The joint log likelihood at each EM step
            can be obtained from the member variable ln_l_array
        """
        self.e_step()
        self.ln_l_array = [self.get_em_objective()]
        for i in range(self.n_em):
            print("step", i)
            #do EM
            self.m_step(self.ln_l_array[len(self.ln_l_array)-1])
            self.e_step()
            #save the log likelihood
            self.ln_l_array.append(self.get_em_objective())
            #check if the log likelihood has decreased small enough
            if self.has_converge(self.ln_l_array):
                break
    
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
            for i in range(len(self)):
                for parameter in self.cp_parameter_array:
                    parameter.calculate_d_reg_self_i(i)
            for parameter in self.cp_parameter_array:
                #do gradient descent
                parameter.gradient_descent(self.step_size)
            #work out log likelihood and test for convergence
            self.update_all_cp_parameters()
            ln_l_array.append(self.get_em_objective())
            if self.has_converge(ln_l_array):
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

class TimeSeriesSgd(TimeSeriesGd):
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
        _permutation_iter: iterator for the permutation of index
    """
    
    def __init__(self, 
                 x,
                 rainfall=None,
                 poisson_rate_n_arma=None,
                 gamma_mean_n_arma=None,
                 cp_parameter_array=None):
        super().__init__(x,
                         rainfall,
                         poisson_rate_n_arma,
                         gamma_mean_n_arma,
                         cp_parameter_array)
        self.n_initial = 100
        self.stochastic_step_size = 0.01
        self.n_stochastic_step = 10
        self.ln_l_max_index = 0
        self.ln_l_stochastic_index = [1]
        self._permutation_iter = self.rng.permutation(len(self)).__iter__()
    
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
                cp_parameter_array = self.copy_parameter()
            #do stochastic gradient descent to get a different initial value
            if i < self.n_initial-1:
                print("stochastic gradient descent")
                #track when stochastic gradient descent was done for this entry
                    #of ln_l_array
                self.ln_l_stochastic_index.append(len(ln_l_all_array))
                for j in range(self.n_stochastic_step):
                    print("step", j)
                    self.m_stochastic_step()
                    self.update_all_cp_parameters()
                    ln_l_all_array.append(self.get_em_objective())
                #track when gradient descent was done
                #the E step right after this in super().fit() is considered part
                    #of stochastic gradient descent
                self.ln_l_stochastic_index.append(len(ln_l_all_array)+1)
            else:
                self.ln_l_stochastic_index.append(len(ln_l_all_array))
        #copy results to the member variable
        self.ln_l_array = ln_l_all_array
        self.set_parameter(cp_parameter_array)
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
            parameter.stochastic_gradient_descent(
                index, self.stochastic_step_size)
    
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
            self._permutation_iter = self.rng.permutation(len(self)).__iter__()
            return self._permutation_iter.__next__()
