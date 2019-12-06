import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import statsmodels.tsa.stattools as stats

def simulate(poisson_rate, gamma_shape, scale, rng):
    y = rng.poisson(poisson_rate)
    x = rng.gamma(gamma_shape * y, scale=scale)
    return(x)

rng = random.RandomState(np.uint32(372625177))
n = 10000
x = np.zeros(n)
for i in range(n):
    x[i] = 0.8*math.sin(2*math.pi/365 * i) + rng.normal()

parameter_0 = (0.088, 0.033, 0.03, 0.055)
parameter_1 = (0.066, 0.079, 0, 0.349)
parameter_2 = (0.077, 0.025, 0.001, 0.873)

y = np.zeros(n)
poisson_rate_array = np.zeros(n)
poisson_rate_array[0] = math.exp(parameter_0[0] * x[0] + parameter_0[3])
gamma_shape_array = np.zeros(n)
gamma_shape_array[0] = math.exp(parameter_1[0] * x[0] + parameter_1[3])
scale_array = np.zeros(n)
scale_array[0] = math.exp(parameter_2[0] * x[0] + parameter_2[3])
y[0] = simulate(poisson_rate_array[0], gamma_shape_array[0], scale_array[0],
    rng)

for i in range(1,n):
    poisson_rate_array[i] = math.exp(parameter_0[0] * x[i]
        + parameter_0[1] * math.log(poisson_rate_array[i-1])
        + parameter_0[2] * y[i-1]
        + parameter_0[3])
    gamma_shape_array[i] = math.exp(parameter_1[0] * x[i] +
        + parameter_1[1] * math.log(gamma_shape_array[i-1])
        + parameter_1[2] * y[i-1]
        + parameter_1[3])
    scale_array[i] = math.exp(parameter_2[0] * x[i] +
        + parameter_2[1] * math.log(scale_array[i-1])
        + parameter_2[2] * y[i-1]
        + parameter_2[3])
    y[i] = simulate(poisson_rate_array[i], gamma_shape_array[i], scale_array[i],
        rng)
        
acf = stats.acf(y, nlags=100, fft=True)
pacf = stats.pacf(y, nlags=10)

plt.figure()
plt.plot(y)
plt.xlabel("Time (day)")
plt.ylabel("Rainfall (mm)")
plt.savefig("../figures/simulation_rain.png")
plt.close()

plt.figure()
plt.plot(x)
plt.xlabel("Time (day)")
plt.ylabel("Model field")
plt.savefig("../figures/simulation_model_field.png")
plt.close()

plt.figure()
plt.bar(np.asarray(range(acf.size)), acf)
plt.xlabel("Time (day)")
plt.ylabel("Autocorrelation")
plt.savefig("../figures/simulation_acf.png")
plt.close()

plt.figure()
plt.bar(np.asarray(range(pacf.size)), pacf)
plt.xlabel("Time (day)")
plt.ylabel("Partial autocorrelation")
plt.savefig("../figures/simulation_pacf.png")
plt.close()

plt.figure()
plt.plot(poisson_rate_array)
plt.xlabel("Time (day)")
plt.ylabel("Poisson rate")
plt.savefig("../figures/simulation_lambda.png")
plt.close()

plt.figure()
plt.plot(gamma_shape_array)
plt.xlabel("Time (day)")
plt.ylabel("Gamma shape")
plt.savefig("../figures/simulation_alpha.png")
plt.close()

plt.figure()
plt.plot(scale_array)
plt.xlabel("Time (day)")
plt.ylabel("Scale")
plt.savefig("../figures/simulation_gamma.png")
plt.close()
