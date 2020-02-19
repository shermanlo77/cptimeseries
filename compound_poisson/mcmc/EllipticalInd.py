import math

from .Mcmc import Mcmc

class EllipticalInd(Mcmc):
    
    def __init__(self, target, rng):
        super().__init__(target, rng)
        self.n_reject_array = []
    
    def sample(self):
        """Uses elliptical slice sampling 
        
        See Murray, Adams, MacKay (2010)
        """
        target = self.target
        state_before = self.state.copy()
        #sample from the prior (with zero mean for now)
        prior_sample = self.simulate_from_prior()
        #sample the vertical line
        ln_y = self.get_log_likelihood() + math.log(self.rng.rand())
        #sample from where on the ellipse
        theta = self.rng.uniform(0, 2 * math.pi)
        edges = [theta - 2 * math.pi, theta]
        
        #keep sampling until one is accepted
        is_sampling = True
        n_reject = 0
        while is_sampling:
            #get a sample (theta = 0 would just sample itself)
            #centre the parameter at zero mean (relative to the prior)
            self.state = ((state_before - target.prior_mean) * math.cos(theta)
                + prior_sample * math.sin(theta))
            #re-centre the parameter
            self.state += target.prior_mean
            #set the new proposed parameter
            #attempt to update all the parameters, reject if there are any
                #numerical problems or when the log likelihood is not large
                #enough
            try:
                self.update_state()
                if self.get_log_likelihood() > ln_y:
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
                n_reject += 1
        self.n_reject_array.append(n_reject)
    
    def simulate_from_prior(self):
        """Return a parameter sampled from the prior
        """
        target = self.target
        prior_sample = self.rng.normal(target.prior_mean, target.prior_cov_chol)
        return prior_sample
