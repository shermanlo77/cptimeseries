import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import statsmodels.tsa.stattools as stats

def simulate(poisson_rate, mean_gamma, dispersion, rng):
    z = rng.poisson(poisson_rate)
    shape = z * dispersion
    if z > 0:
        scale = mean_gamma/shape
    else:
        scale = 0
    y = rng.gamma(shape, scale=scale)
    return(y, z)

rng = random.RandomState(np.uint32(372625180))
n = 10000
x = np.zeros(n)
for i in range(n):
    x[i] = 0.8*math.sin(2*math.pi/365 * i) + rng.normal()

parameter_0 = (0.098, 0.713, 0.01, 0.055) #poisson
parameter_1 = (0.066, 0.46, 0.09, 0.8) #mean
parameter_2 = (0.07, 0.373) #dispersion

y = np.zeros(n)
z = np.zeros(n)
poisson_rate_array = np.zeros(n)
poisson_rate_array[0] = math.exp(parameter_0[0] * x[0] + parameter_0[3])
mean_gamma_array = np.zeros(n)
mean_gamma_array[0] = math.exp(parameter_1[0] * x[0] + parameter_1[3])
dispersion_array = np.zeros(n)
dispersion_array[0] = math.exp(parameter_2[0] * x[0] + parameter_2[1])
y[0], z[0] = simulate(poisson_rate_array[0], mean_gamma_array[0],
    dispersion_array[0], rng)

for i in range(1,n):
    
    poisson_rate_array[i] = math.exp(parameter_0[0] * x[i]
        + parameter_0[1] * math.log(poisson_rate_array[i-1])
        + parameter_0[2] * (z[i-1] - poisson_rate_array[i-1])
        + parameter_0[3])
    
    exponent = parameter_1[0] * x[i] \
        + parameter_1[1] *  math.log(mean_gamma_array[i-1]) \
        + parameter_1[3]
    if z[i-1] > 0:
        exponent +=\
            parameter_1[2] * (y[i-1]/z[i-1] - mean_gamma_array[i-1])
    mean_gamma_array[i] = math.exp(exponent)
    
    dispersion_array[i] = math.exp(parameter_2[0] * x[i]
        + parameter_2[1])
    
    y[i], z[i] = simulate(poisson_rate_array[i], mean_gamma_array[i],
        dispersion_array[i], rng)

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
plt.plot(mean_gamma_array)
plt.xlabel("Time (day)")
plt.ylabel("Mean of gamma")
plt.savefig("../figures/simulation_mu.png")
plt.show()
plt.close()

plt.figure()
plt.plot(dispersion_array)
plt.xlabel("Time (day)")
plt.ylabel("Dispersion")
plt.savefig("../figures/simulation_dispersion.png")
plt.show()
plt.close()
