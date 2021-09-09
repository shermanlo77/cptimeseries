# a quick script to bootstrap the forecasts and suggest the number of decimial
# places to use for the area uncer the ROC curve

import math

import joblib
import numpy as np
from numpy import random
import pandas as pd

import dataset


def main():
    time_series = joblib.load("result/TimeSeriesHyperSlice.gz")
    test_set = dataset.CardiffTest()
    test_rain = test_set.rain

    forecaster = time_series.forecaster
    forecaster.load_memmap("r")

    seed = random.SeedSequence(254267254235771235840594891069714545013)
    rng = random.RandomState(random.MT19937(seed))

    rain_array = [0, 5, 10, 15, 20, 25, 30]
    decimial_place_array = []
    n_bootstrap = 32

    for rain in rain_array:
        auc_array = []
        for i in range(n_bootstrap):
            bootstrap = forecaster.bootstrap(rng)
            roc = bootstrap.get_roc_curve(rain, test_rain)
            auc_array.append(roc.area_under_curve)
        auc_std = np.std(auc_array, ddof=1)
        decimial_place_array.append(-round(math.log10(auc_std)))

    data_frame = pd.DataFrame(
        decimial_place_array, rain_array, ["no. dec. places"])
    print("rain (mm)")
    print(data_frame)


if __name__ == "__main__":
    main()
