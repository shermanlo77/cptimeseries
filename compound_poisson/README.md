# Compound Poisson
* Copyright (c) 2020 Sherman Lo
* MIT LICENSE

Under the hood code for Bayesian inference for spatial-temporal compound-Poisson data.

## Sub-packages
* `fit`
    * Wrapper classes for fitting the model onto data
* `forecast`
    * Classes for conducting forecasts given the fitted model
* `mcmc`
    * Implementations of MCMC algorithms and target distributions

## Description of code
* `time_series.py` contains the base class `TimeSeries` used for single location time series. It uses the modules `terms.py`, which handles compound poisson terms and sums, `arma.py`, which handles ARMA terms for training and forecasting, and `parameter.py`, which handles the compound poisson parameters which varies with time.
* `time_series_gradient.py` contains old and unused code for a frequentist approach to inference using gradient and stochastic gradient descent.
* `time_series_mcmc.py` contains classes for Bayesian inference, using Gibbs sampling, for the time series. `TimeSeriesMcmc` uses a Metropolis-Hastings with Gibbs approach. `TimeSeriesSlice` uses slice and elliptical slice sampling. `TimeSeriesHyperSlice` introduce a hyper parameter for the variance term, inferred using Metropolis-Hastings.
* `downscale.py` implements multiple time series by imposing a Gaussian process prior. `Downscale` impose a GP prior on the parameters. `DownscaleDual`, in addition, impose a GP prior on the model fields on the coarse grid. An additional Gibbs step was implemented to sample the model fields on the fine grid.
  * Developers note: `TimeSeriesDownscale` and `TimeSeriesDownscaleDual` are subclasses of `TimeSeriesSlice` and are used in the `Downscale` family of classes.
* `print.py` contains functions for plotting figures for `TimeSeries` and `Forecaster` objects.
* `roc.py` contain a class for plotting ROC curves. Used by `print.py` and `forecast`.
* `multiprocess.py` contain wrapper classes for multi-thread work. Adjusting the instantiation of objects for multi-thread work can be done here. Please refer to the manual of the corresponding package when using MPI:
    * [mpi4py.futures](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html)
    * [abcpy.backends](https://abcpy.readthedocs.io/en/v0.5.7/parallelization.html)
