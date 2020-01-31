import math
import numpy as np

from TimeSeriesSlice import TimeSeriesSlice

class TimeSeriesElliptical(TimeSeriesSlice):
    
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
    
    def sample_reg(self):
        """Override - Uses elliptical slice sampling 
        
        See Murray, Adams, MacKay (2010)
        """
        
        parameter_before = self.get_parameter_vector()
        #sample from the prior (with zero mean for now)
        prior_sample = self.rng.multivariate_normal(
            np.zeros(self.n_parameter), self.prior_covariance)
        #sample the vertical line
        ln_y = self.get_joint_log_likelihood() + math.log(self.rng.rand())
        #sample from where on the ellipse
        theta = self.rng.uniform(0, 2 * math.pi)
        edges = [theta - 2 * math.pi, theta]
        
        #keep sampling until one is accepted
        is_sampling = True
        while is_sampling:
            #get a sample (theta = 0 would just sample itself)
            #centre the parameter at zero mean (relative to the prior)
            parameter = ((parameter_before - self.prior_mean) * math.cos(theta)
                + prior_sample * math.sin(theta))
            #re-centre the parameter
            parameter += self.prior_mean
            #set the new proposed parameter
            self.set_parameter_vector(parameter)
            #attempt to update all the parameters, reject if there are any
                #numerical problems or when the log likelihood is not large
                #enough
            try:
                self.update_all_cp_parameters()
                if self.get_joint_log_likelihood() > ln_y:
                    is_accept = True
                else:
                    is_accept = False
            except(ValueError, OverflowError):
                is_accept = False
            #stop sampling if to accept the new proposed parameter
            #change the search space if the new proposed parameter was rejected
            if is_accept:
                is_sampling = False
            else:
                if theta < 0:
                    edges[0] = theta
                else:
                    edges[1] = theta
                theta =  self.rng.uniform(edges[0], edges[1])
            
