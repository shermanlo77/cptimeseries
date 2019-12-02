import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import statsmodels.tsa.stattools as stats

def simulate(poisson_rate, gamma_shape, scale):
    y = random.poisson(poisson_rate)
    x = random.gamma(gamma_shape * y, scale=scale)
    return(x)

n = 10000
x = np.zeros(n)
for i in range(n):
    x[i] = 0.8*math.sin(2*math.pi/365 * i) + random.normal()

parameter_0 = (0.088, 0.033, 0.265)
parameter_1 = (0.066, 0.079, 0.049)
parameter_2 = (0.077, 0.025, 1.573)

poisson_rate_array = np.zeros(n)
poisson_rate_array[0] = math.exp(parameter_0[0] * x[0] + parameter_0[2])
for i in range(1,n):
    poisson_rate_array[i] = math.exp(parameter_0[0] * x[i] +
        parameter_0[1] * x[i-1] + parameter_0[2])

gamma_shape_array = np.zeros(n)
gamma_shape_array[0] = math.exp(parameter_1[0] * x[0] + parameter_1[2])
for i in range(1,n):
    gamma_shape_array[i] = math.exp(parameter_1[0] * x[i] +
        parameter_1[1] * x[i-1] + parameter_1[2])

scale_array = np.zeros(n)
scale_array[0] = math.pow(parameter_2[0] * x[0] + parameter_2[2], 2)
for i in range(1,n):
    scale_array[i] = math.pow(parameter_2[0] * x[i] +
        parameter_2[1] * x[i-1] + parameter_2[2], 2)

y = simulate(poisson_rate_array, gamma_shape_array, scale_array)
acf = stats.acf(y, nlags=150, fft=True)
pacf = stats.pacf(y, nlags=10)

plt.figure()
plt.plot(y)
plt.xlabel("Time (day)")
plt.ylabel("Rainfall (mm)")
plt.show()

plt.figure()
plt.bar(np.asarray(range(acf.size)), acf)
plt.xlabel("Lag (day)")
plt.ylabel("Autocorrelation")
plt.show()

plt.figure()
plt.bar(np.asarray(range(pacf.size)), pacf)
plt.xlabel("Lag (day)")
plt.ylabel("Partial autocorrelation")
plt.show()

plt.figure()
plt.plot(poisson_rate_array)
plt.xlabel("Lag (day)")
plt.ylabel("Poisson rate")
plt.show()

plt.figure()
plt.plot(gamma_shape_array)
plt.xlabel("Lag (day)")
plt.ylabel("Gamma shape")
plt.show()

plt.figure()
plt.plot(scale_array)
plt.xlabel("Lag (day)")
plt.ylabel("Sale")
plt.show()
