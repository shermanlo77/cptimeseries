import compound_poisson as cp
import matplotlib.pyplot as plot
import numpy as np
import joblib
import math
import numpy.random as random
from Arma import Arma
from simulate import simulate_training, simulate_test

time_series = joblib.load("mcmc_result.zlib")
n = time_series.n

n_burnin = 60000
n_sample = 100

"""TO BE DELETED"""
time_series.fitted_time_series = None
for parameter in time_series.cp_parameter_array:
    parameter.arma = Arma(parameter)

rng = random.RandomState(np.uint32(292203111))
y_prediction = []
rmse_array = np.zeros(n_sample)
deviance_array = np.zeros(n_sample)
for i in range(n_sample):
    print("Predictive sample", i)
    time_series.set_parameter_from_sample(
        rng.randint(n_burnin, time_series.n_sample))
    forecast = time_series.self_forecast_simulate(rng)
    y_prediction.append(forecast.y_array)
    rmse_array[i] = forecast.get_error_rmse(time_series.y_array)
    deviance_array[i] = forecast.get_error_square_sqrt(time_series.y_array)

y_prediction_mean = np.mean(np.asarray(y_prediction), 0)
forecast.y_array = y_prediction_mean
print("root mean square =",
      forecast.get_error_rmse(time_series.y_array),
      "mm")
print("normalised deviance =",
      forecast.get_error_square_sqrt(time_series.y_array),
      "mm")


plot.figure()
plot.plot(time_series.y_array)
plot.ylabel("rainfall (mm)")
plot.xlabel("time (day)")
plot.savefig("../figures/forecast_simulation/self_true.eps")
plot.show()
plot.close()

plot.figure()
plot.plot()
plot.plot(y_prediction[0])
plot.ylabel("rainfall (mm)")
plot.xlabel("time (day)")
plot.savefig("../figures/forecast_simulation/self_y_sample.eps")
plot.show()
plot.close()

plot.figure()
plot.plot(time_series.y_array)
plot.plot(y_prediction_mean)
plot.ylabel("rainfall (mm)")
plot.xlabel("time (day)")
plot.savefig("../figures/forecast_simulation/self_y_mean.eps")
plot.show()
plot.close()

plot.figure()
plot.plot(y_prediction_mean - time_series.y_array)
plot.ylabel("residual (mm)")
plot.xlabel("time (day)")
plot.savefig("../figures/forecast_simulation/self_residual.eps")
plot.show()
plot.close()

plot.figure()
plot.hist(rmse_array)
plot.xlabel("root mean square error (mm)")
plot.savefig("../figures/forecast_simulation/self_rmse.eps")
plot.show()
plot.close()

plot.figure()
plot.hist(deviance_array)
plot.xlabel("normalised deviance (mm)")
plot.savefig("../figures/forecast_simulation/self_deviance.eps")
plot.show()
plot.close()
