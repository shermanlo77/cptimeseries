# compound_poisson
* Copyright (c) 2020 Sherman Lo
* MIT LICENSE

Simulation, Bayesian inference and forecasting of precipitation using the compound Poisson model.

## Packages
* `compound_poisson.fit`
    * Modules for fitting the model onto data.
* `compound_poisson.forecast`
    * Modules for forward simulation given the fitted model. Also contain modules for assessing the performance of the forecast. Please refer to the README as well.
* `compound_poisson.mcmc`
    * Implementations of MCMC algorithms and target distributions.

## Single Location Time Series
`compound_poisson.time_series.TimeSeries` &larr; `compound_poisson.time_series_mcmc.TimeSeriesMcmc` &larr; `compound_poisson.time_series_mcmc.TimeSeriesSlice` &larr; `compound_poisson.time_series_mcmc.TimeSeriesHyperSlice`

* The base class is `compound_poisson.time_series.TimeSeries` and is used for single location time series. It uses the following modules:
    * `compound_poisson.terms` handles compound Poisson terms and sum,
    * `compound_poisson.arma` handles ARMA terms for training and forecasting,
    * `compound_poisson.parameter` handles the compound Poisson parameters which varies with time.
* The module `compound_poisson.time_series_mcmc` contains classes for Bayesian inference, using Gibbs sampling, for the time series.
    * `compound_poisson.time_series.TimeSeriesMcmc` uses a full Metropolis-Hastings within Gibbs approach,
    * `compound_poisson.time_series_mcmc.TimeSeriesSlice` uses slice and elliptical slice sampling,
    * `compound_poisson.time_series_mcmc.TimeSeriesHyperSlice` introduces a hyper parameter for the variance term, inferred using Metropolis-Hastings.

## Multiple Location Time Series
`compound_poisson.downscale.Downscale` &#x25C7;-1..\* `compound_poisson.downscale.TimeSeriesDownscale`
 `compound_poisson.time_series_mcmc.TimeSeriesSlice` &larr; `compound_poisson.downscale.TimeSeriesDownscale`

* The class `compound_poisson.downscale.Downscale` implements multiple time series. It imposes a Gaussian process prior on the parameters.

## Notes for Developers
* The module `compound_poisson.multiprocess` contain wrapper classes for multi-thread work. Adjusting the instantiation, thread joining and destruction of objects for multi-thread work can be done here. Please refer to the manual of the corresponding package when using MPI:
    * [mpi4py.futures](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html)
    * [abcpy.backends](https://abcpy.readthedocs.io/en/v0.5.7/parallelization.html)
