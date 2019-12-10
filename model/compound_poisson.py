import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import statsmodels.tsa.stattools as stats

class Parameters:
    """Contains regressive, autoregressive, moving average and constant parameters
        for the compound Poisson time series model
    
    Attributes:
        poisson_rate: dictionary with keys "reg", "AR", "MA", "const" with
            corresponding values regression parameters in the poisson rate
        gamma_mean: dictionary with keys "reg", "AR", "MA", "const" with
            corresponding values regression parameters in the gamma mean
        gamma_dispersion: dictionary with keys "reg", "const" which
            corresponding values regression parameters in the gamma dispersion
    
    """
    
    def __init__(self, n_dim):
        """
        Args:
            n_dim:
        """
        self.n_dim = n_dim
        self.poisson_rate = {
            "reg": np.zeros(n_dim),
            "AR": 0.0,
            "MA": 0.0,
            "const": 0.0,
        }
        self.gamma_mean = {
            "reg": np.zeros(n_dim),
            "AR": 0.0,
            "MA": 0.0,
            "const": 0.0,
        }
        self.gamma_dispersion = {
            "reg": np.zeros(n_dim),
            "const": 0.0,
        }

#CLASS: COMPOUND POISSON TIME SERIES
class CompoundPoissonTimeSeries:
    """Compound Poisson Time Series with ARMA behaviour
    
    Attributes:
        x: design matrix of the model fields, shape (n, n_dim)
        n: length of time series
        n_dim: number of dimensions of the model fields
        parameters: Parameters object
        poisson_rate_array: poisson rate at each time step
        gamma_mean_array: gamma mean at each time step
        gamma_dispersion_array: gamma dispersion at each time step
        z: latent poisson variables at each time step
        y: compound poisson variables at each time step
    """
    
    def __init__(self, x, parameters):
        """
        Args:
            x: design matrix of the model fields, shape (n, n_dim)
            parameters: Parameters object
        """
        self.x = x
        self.n = x.shape[0]
        self.n_dim = x.shape[1]
        self.parameters = parameters
        self.poisson_rate_array = np.zeros(self.n)
        self.gamma_mean_array = np.zeros(self.n)
        self.gamma_dispersion_array = np.zeros(self.n)
        self.z = np.zeros(self.n)
        self.y = np.zeros(self.n)
    
    def simulate(self, rng):
        """Simulate a whole time series
        
        Simulate a time series given the model fields self.x, parameters
            and self.parameters. Modify the member variables poisson_rate_array,
            gamma_mean_array and gamma_dispersion_array with the parameters at
            each time step. Also modifies self.z and self.y with the simulated 
            values.
        
        Args:
            rng: numpy.random.RandomState object
        """
        
        #simulate n times
        for i in range(self.n):
            
            #regressive on the model fields and constant
            self.poisson_rate_array[i] = (
                np.dot(self.parameters.poisson_rate["reg"], self.x[i,:])
                + self.parameters.poisson_rate["const"])
            self.gamma_mean_array[i] = (
                np.dot(self.parameters.gamma_mean["reg"], self.x[i,:])
                + self.parameters.gamma_mean["const"])
            self.gamma_dispersion_array[i] = (
                np.dot(self.parameters.gamma_dispersion["reg"], self.x[i,:])
                + self.parameters.gamma_dispersion["const"])
            
            #for the second step and beyond, add AR and MA terms
            #the AR terms regress on its log self (subtracted from itself when
                #all the parameters are zero)
            #the MA terms regress on the difference of the previous value of the
                #variable with the expected value, normalised to the sqrt
                #variance of the variable
            if i > 0:
                
                #poisson rate
                self.poisson_rate_array[i] += (
                    self.parameters.poisson_rate["AR"]
                    *(math.log(self.poisson_rate_array[i-1])
                    - self.parameters.poisson_rate["const"]))
                self.poisson_rate_array[i] += (
                    self.parameters.poisson_rate["MA"]
                    * (self.z[i-1] - self.poisson_rate_array[i-1])
                    / math.sqrt(self.poisson_rate_array[i-1]))
                
                #gamma mean
                self.gamma_mean_array[i] += (
                    self.parameters.gamma_mean["AR"]
                    * (math.log(self.gamma_mean_array[i-1])
                    - self.parameters.gamma_mean["const"]))
                #when z is zero, y is zero, giving an error term of zero
                    #therefore, do not add any z=0 terms
                if self.z[i-1] > 0:
                    self.gamma_mean_array[i] += (
                        self.parameters.gamma_mean["MA"]
                        * (self.y[i-1]/self.z[i-1]- self.gamma_mean_array[i-1])
                        / self.gamma_mean_array[i-1]
                        / math.sqrt(
                            self.gamma_dispersion_array[i-1]
                            * self.z[i-1])
                    )
            
            #exp link function, make it positive
            self.poisson_rate_array[i] = math.exp(self.poisson_rate_array[i])
            self.gamma_mean_array[i] = math.exp(self.gamma_mean_array[i])
            self.gamma_dispersion_array[i] = (
                math.exp(self.gamma_dispersion_array[i]))
            
            #simulate this compound Poisson
            self.y[i], self.z[i] = simulate_cp(
                self.poisson_rate_array[i], self.gamma_mean_array[i],
                self.gamma_dispersion_array[i], rng)


def simulate_cp(poisson_rate, gamma_mean, gamma_dispersion, rng):
    """Simulate a single compound poisson random variable
    
    Args:
        poisson_rate: scalar
        gamma_mean: scalar
        gamma_dispersion: scalar
        rng: numpy.random.RandomState object
    
    Returns:
        y: compound Poisson random variable
        z: latent Poisson random variable
    """
    
    z = rng.poisson(poisson_rate) #poisson random variable
    shape = z / gamma_dispersion #gamma shape parameter
    #if z is zero, set the parameters of the gamma distribution to be zero
    if z > 0:
        scale = gamma_mean/shape
    else:
        scale = 0
    y = rng.gamma(shape, scale=scale) #gamma random variable
    return(y, z)

def main():
    
    rng = random.RandomState(np.uint32(372625178))
    n = 10000
    x = np.zeros((n,1))
    for i in range(n):
        x[i] = 0.8*math.sin(2*math.pi/365 * i) + rng.normal()
    
    parameters = Parameters(1)
    parameters.poisson_rate["reg"] = 0.098
    parameters.poisson_rate["AR"] = 0.713
    parameters.poisson_rate["MA"] = 0.11
    parameters.poisson_rate["const"] = 0.355
    parameters.gamma_mean["reg"] = 0.066
    parameters.gamma_mean["AR"] = 0.4
    parameters.gamma_mean["MA"] = 0.24
    parameters.gamma_mean["const"] = 1.3
    parameters.gamma_dispersion["reg"] = 0.07
    parameters.gamma_dispersion["const"] = 0.373
    
    compound_poisson_time_series = CompoundPoissonTimeSeries(x, parameters)
    compound_poisson_time_series.simulate(rng)
    y = compound_poisson_time_series.y
    poisson_rate_array = compound_poisson_time_series.poisson_rate_array
    gamma_mean_array = compound_poisson_time_series.gamma_mean_array
    gamma_dispersion_array = compound_poisson_time_series.gamma_dispersion_array
    
    acf = stats.acf(y, nlags=100, fft=True)
    pacf = stats.pacf(y, nlags=10)

    plt.figure()
    plt.plot(y)
    plt.xlabel("Time (day)")
    plt.ylabel("Rainfall (mm)")
    plt.savefig("../figures/simulation_rain.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(x)
    plt.xlabel("Time (day)")
    plt.ylabel("Model field")
    plt.savefig("../figures/simulation_model_field.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.bar(np.asarray(range(acf.size)), acf)
    plt.xlabel("Time (day)")
    plt.ylabel("Autocorrelation")
    plt.savefig("../figures/simulation_acf.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.bar(np.asarray(range(pacf.size)), pacf)
    plt.xlabel("Time (day)")
    plt.ylabel("Partial autocorrelation")
    plt.savefig("../figures/simulation_pacf.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(poisson_rate_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Poisson rate")
    plt.savefig("../figures/simulation_lambda.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(gamma_mean_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Mean of gamma")
    plt.savefig("../figures/simulation_mu.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(gamma_dispersion_array)
    plt.xlabel("Time (day)")
    plt.ylabel("Dispersion")
    plt.savefig("../figures/simulation_dispersion.png")
    plt.show()
    plt.close()

main()
