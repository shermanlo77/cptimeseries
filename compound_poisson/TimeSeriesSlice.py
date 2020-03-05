from .TimeSeriesMcmc import TimeSeriesMcmc
from .mcmc import Elliptical, TargetParameter, TargetZ, ZSlice

class TimeSeriesSlice(TimeSeriesMcmc):
    
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
        self.parameter_mcmc = Elliptical(self.parameter_target, self.rng)
        self.z_mcmc = ZSlice(self.z_target, self.rng)
    
