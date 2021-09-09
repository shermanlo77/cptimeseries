from compound_poisson import fit
from compound_poisson.fit import wrapper
import dataset


def main():
    fitter = fit.time_series.FitterHyperSlice()
    training = dataset.Cardiff10Training()
    test = dataset.CardiffTest()
    default_burn_in = 30000
    wrapper.time_series_forecast(fitter, training, test, default_burn_in)


if __name__ == "__main__":
    main()
