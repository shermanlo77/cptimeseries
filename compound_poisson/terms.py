import math
import numpy as np
from scipy import special

class Terms(object):
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
        self.y = parent[index]
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
                if log_ratio > Terms.cp_sum_threshold:
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
            if log_ratio > Terms.cp_sum_threshold:
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
        terms[2] = -special.loggamma(z/self.gamma_dispersion)
        terms[3] = z*math.log(self.y)/self.gamma_dispersion
        terms[4] = z*math.log(self.poisson_rate)
        terms[5] = -special.loggamma(1+z)
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
