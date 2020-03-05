from .mcmc import Rwmh, TargetPrecision
from .TimeSeriesSlice import TimeSeriesSlice

class TimeSeriesHyperSlice(TimeSeriesSlice):
    """Fit Compound Poisson time series using slice sampling with a prior on the
    precision
    
    Method uses slice within Gibbs. Uniform prior on z, Normal prior on the
        regression parameters, Gamma prior on the precision of the covariance of
        the Normal prior. Gibbs sample either z, regression parameters or the
        precision. Sample z using slice sampling (Neal, 2003). Sampling the
        parameters using elliptical slice sampling (Murray 2010).
    
    For more attributes, see the superclass
    Attributes:
        precision_target: wrapper Target object with evaluates the posterior of
            the precision
        precision_mcmc: Mcmc object for precision_target
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
        self.precision_target = TargetPrecision(self.parameter_target)
        #mcmc object evaluated at instantiate_mcmc
        self.precision_mcmc = None
    
    def fit(self):
        """Fit using Gibbs sampling
        
        Override - Gibbs sample either z, regression parameters or the
            precision. 
        """
        self.initalise_z()
        self.instantiate_mcmc()
        self.update_precision()
        #initial value is a sample
        self.parameter_mcmc.add_to_sample()
        self.z_mcmc.add_to_sample()
        self.precision_mcmc.add_to_sample()
        #Gibbs sampling
        for i in range(self.n_sample):
            print("Sample",i)
            #select random component
            rand = self.rng.rand()
            if rand < 1/3:
                self.z_mcmc.step()
                self.parameter_mcmc.add_to_sample()
                self.precision_mcmc.add_to_sample()
            elif rand < 2/3:
                self.parameter_mcmc.step()
                self.z_mcmc.add_to_sample()
                self.precision_mcmc.add_to_sample()
            else:
                self.precision_mcmc.step()
                self.parameter_mcmc.add_to_sample()
                self.z_mcmc.add_to_sample()
                #update the prior covariance in parameter_target
                self.update_precision()
    
    def update_precision(self):
        """Propagate the precision in precision_target to parameter_target
        """
        self.parameter_target.prior_cov_chol = (
            self.precision_target.get_cov_chol())
    
    def instantiate_mcmc(self):
        """Instantiate all MCMC objects
        
        Override - instantiate the MCMC for the precision
        """
        super().instantiate_mcmc()
        self.precision_mcmc = Rwmh(self.precision_target, self.rng)
        self.precision_mcmc.proposal_covariance_small = 1e-4
    
    def simulate_parameter_from_prior(self):
        """Simulate parameter from the prior
        
        Override - Sample the precision from the prior and then sample the
            parameter from the prior
        """
        prior_precision = self.precision_target.simulate_from_prior(self.rng)
        self.update_precision()
        return super().simulate_parameter_from_prior()
