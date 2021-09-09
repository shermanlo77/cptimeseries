from compound_poisson import fit
from compound_poisson.fit import wrapper
import dataset


def main():
    fitter = fit.time_series.FitterHyperSlice()
    training = dataset.Cardiff1Training()
    test = dataset.CardiffTest()
    default_burn_in = 10000
    wrapper.time_series_forecast(fitter, training, test, default_burn_in)


if __name__ == "__main__":
    main()
