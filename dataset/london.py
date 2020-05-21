import numpy as np
from numpy import random
import pandas as pd

import compound_poisson
from compound_poisson import parameter
import dataset
from dataset import location

class LondonTraining(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDual10Training(), "London")

class LondonTest(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDual10Test(), "London")

class LondonSimulatedTraining(LondonTraining):

    def __init__(self):
        self.time_series = None
        super().__init__()

    def copy_from(self, other):
        super().copy_from(other)
        self.time_series = other.time_series

    def load_data(self):
        super().load_data()
        time_series, training_slice, test_slice = get_simulated_time_series()
        self.time_series = time_series
        self.rain = self.time_series.y_array[training_slice]

def get_simulated_time_series():
    """Return a simulated time_series object and 2 slice objec

    Returna 3 tuple.
        -TimeSeries object which has simulated values throughout the whole of
            the training and test set
        -slice object indicating the training set
        -slice object indicating the test set
    """

    #concatenate the model fields together so that both the training and test
        #times are simulated
    data_training = LondonTraining()
    data_test = LondonTest()
    model_field = pd.concat(
        [data_training.get_model_field(), data_test.get_model_field()])

    training_slice = slice(0, len(data_training))
    test_slice = slice(len(data_training), len(data_training)+len(data_test))

    rng = random.RandomState(np.uint32(3667413888))
    n_model_field = len(model_field.columns)
    n_arma = (5, 5)
    poisson_rate = parameter.PoissonRate(n_model_field, n_arma)
    gamma_mean = parameter.GammaMean(n_model_field, n_arma)
    gamma_dispersion = parameter.GammaDispersion(n_model_field)
    cp_parameter_array = [poisson_rate, gamma_mean, gamma_dispersion]
    poisson_rate['reg'] = np.asarray([
        -0.11154721,
        0.01634086,
        0.45595715,
        -0.3993777,
        0.09398412,
        -0.22794538,
        0.07249126,
        -0.21600272,
        -0.05372614,
    ])
    poisson_rate['const'] = np.asarray([-0.92252178])
    poisson_rate['AR'] = np.asarray([
        0.2191157,
        0.0828164,
        -0.08994476,
        0.08133209,
        -0.09344768,
    ])
    poisson_rate['MA'] = np.asarray([
        0.22857258,
        0.16147521,
        -0.02136632,
        0.04896173,
        0.02372191,
    ])
    gamma_mean['reg'] = np.asarray([
        -0.09376735,
        -0.01028988,
        0.02133337,
        0.15878673,
        -0.15329763,
        0.17121309,
        -0.18262059,
        -0.1709044,
        -0.11908832,
    ])
    gamma_mean['const'] = np.asarray([1.18446041])
    gamma_mean['AR'] = np.asarray([
        0.22679054,
        0.10105583,
        -0.05324423,
        0.03245928,
        0.02608218,
    ])
    gamma_mean['MA'] = np.asarray([
        0.21196802,
        0.18057783,
        -0.06592883,
        0.06715984,
        -0.05437931,
    ])
    gamma_dispersion['reg'] = np.asarray([
        0.07291021,
        0.34183881,
        0.20085349,
        0.2210854,
        0.1586696,
        0.37656874,
        0.03970588,
        -0.15201423,
        -0.13569733,
    ])
    gamma_dispersion['const'] = np.asarray([-0.26056112])

    time_series = compound_poisson.TimeSeries(
        model_field, cp_parameter_array=cp_parameter_array)
    time_series.rng = rng
    time_series.simulate()
    time_series.time_array = np.concatenate(
        [data_training.time_array, data_test.time_array]).tolist()

    return (time_series, training_slice, test_slice)
