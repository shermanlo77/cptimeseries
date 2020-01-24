from TimeSeries import TimeSeries

class TimeSeriesGd(TimeSeries):
    """Compound Poisson Time Series with ARMA behaviour
    
    Attributes:
        ln_l_array: the joint log likelihood after calling the method fit()
        step_size: the step size for gradient descent, used in the M step
        n_em: number of EM steps
        n_gradient_descent: number of steps in a M step
        min_ln_l_ratio: determines when to stop the EM algorithm if the log
            likelihood increases not very much
    """
    
    def __init__(self, x, cp_parameter_array=None, rainfall=None):
        super().__init__(x, cp_parameter_array, rainfall)
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
