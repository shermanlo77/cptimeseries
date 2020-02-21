from .mcmc import Rwmh, TargetPrecision
from .TimeSeriesSlice import TimeSeriesSlice

class TimeSeriesHyperSlice(TimeSeriesSlice):
    
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
        self.precision_mcmc = None
    
    def fit(self):
        """Do MCMC
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
                self.update_precision()
    
    def update_precision(self):
        self.parameter_target.prior_cov_chol = (
            self.precision_target.get_cov_chol())
    
    def instantiate_mcmc(self):
        super().instantiate_mcmc()
        self.precision_mcmc = Rwmh(self.precision_target, self.rng)
        self.precision_mcmc.proposal_covariance_small = 1e-4
    
    def simulate_parameter_from_prior(self):
        prior_precision = self.precision_target.simulate_from_prior(self.rng)
        self.update_precision()
        return super().simulate_parameter_from_prior()
