from .TimeSeriesMcmc import TimeSeriesMcmc
from .mcmc import Elliptical, TargetParameter, TargetZ, ZSlice

class TimeSeriesSlice(TimeSeriesMcmc):
    """Fit Compound Poisson time series using slice sampling from a Bayesian
    setting
    
    Method uses slice within Gibbs. Sample either the z or the
        regression parameters. Uniform prior on z, Normal prior on the
        regression parameters. Sample z using slice sampling (Neal, 2003).
        Sampling the parameters using elliptical slice sampling (Murray 2010).
    
    For more attributes, see the superclass
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
        self.n_sample = 10000
    
    def instantiate_mcmc(self):
        """Instantiate all MCMC objects
        
        Override
        Instantiate slice sampling for the parameter and z
        """
        self.parameter_mcmc = Elliptical(self.parameter_target, self.rng)
        self.z_mcmc = ZSlice(self.z_target, self.rng)
    
