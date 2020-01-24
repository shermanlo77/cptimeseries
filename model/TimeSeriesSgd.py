import numpy as np
import numpy.random as random

from TimeSeriesGd import TimeSeriesGd

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
    
    def __init__(self, x, cp_parameter_array=None, rainfall=None):
        super().__init__(x, cp_parameter_array, rainfall)
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
