import compound_poisson as cp
import matplotlib.pyplot as plot
import numpy as np
import joblib
import math
import numpy.random as random
from Arma import Arma
from simulate import simulate_training, simulate_test

time_series = joblib.load("mcmc_result.zlib")

n_burnin = 60000
n_sample = 100

time_series_test = simulate_test()

"""TO BE DELETED"""
time_series.fitted_time_series = None
for parameter in time_series.cp_parameter_array:
    parameter.arma = Arma(parameter)

rng = random.RandomState(np.uint32(877276962))
y_simulation = []
rmse_array = np.zeros(n_sample)
deviance_array = np.zeros(n_sample)
for i in range(n_sample):
    print("Predictive sample", i)
    time_series.set_parameter_from_sample(
        rng.randint(n_burnin, time_series.n_sample))
    simulation = time_series.simulate_future(time_series_test.x, rng)
    y_simulation.append(simulation.y_array)
    forecast = time_series.forecast(time_series_test.x)
    rmse_array[i] = forecast.get_error_rmse(time_series_test.y_array)
    deviance_array[i] = forecast.get_error_square_sqrt(time_series_test.y_array)

y_prediction = np.mean(np.asarray(y_simulation), 0)
forecast.y_array = y_prediction
print("root mean square =",
      forecast.get_error_rmse(time_series_test.y_array),
      "mm")
print("normalised deviance =",
      forecast.get_error_square_sqrt(time_series_test.y_array),
      "mm")

plot.figure()
plot.plot(np.concatenate((time_series.y_array, time_series_test.y_array)))
plot.ylabel("rainfall (mm)")
plot.xlabel("time (day)")
plot.savefig("../figures/forecast_simulation/forecast_true.eps")
plot.show()
plot.close()

plot.figure()
plot.plot(range(time_series.n), time_series.y_array)
plot.plot(range(time_series.n, time_series.n+forecast.n), y_simulation[0])
plot.ylabel("rainfall (mm)")
plot.xlabel("time (day)")
plot.savefig("../figures/forecast_simulation/forecast_y_sample.eps")
plot.show()
plot.close()

plot.figure()
plot.plot(np.concatenate((time_series.y_array, time_series_test.y_array)))
plot.plot(range(time_series.n, time_series.n+forecast.n), y_prediction)
plot.ylabel("rainfall (mm)")
plot.xlabel("time (day)")
plot.savefig("../figures/forecast_simulation/forecast_y_mean.eps")
plot.show()
plot.close()

plot.figure()
plot.plot(range(time_series.n, time_series.n+forecast.n),
          y_prediction - time_series.y_array)
plot.ylabel("residual (mm)")
plot.xlabel("time (day)")
plot.savefig("../figures/forecast_simulation/forecast_residual.eps")
plot.show()
plot.close()

plot.figure()
plot.hist(rmse_array)
plot.xlabel("root mean square error (mm)")
plot.savefig("../figures/forecast_simulation/forecast_rmse.eps")
plot.show()
plot.close()

plot.figure()
plot.hist(deviance_array)
plot.xlabel("normalised deviance (mm)")
plot.savefig("../figures/forecast_simulation/forecast_deviance.eps")
plot.show()
plot.close()
