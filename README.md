# Downscale using Compound Poisson
* Copyright (c) 2020 Sherman Lo

Concept code for predicting precipitation using model fields (temperature, geopotential, wind velocity, etc.) as predictors for sub-areas across the British Isle.

The modification of code is required to use your own data. This is because the code is designed for `.grib` and `.nc` files with specific grids. Please see LICENCE for further information on how you can use this code for your purpose.

Keywords:
* Downscaling
* Compound Poisson
* Gaussian Process
* Markov Chain Monte Carlo
* Time series
* Spatial–temporal

## Requirements (Python 3)
* At least 16 GB of RAM
* `numpy`
* `pandas`
* `scipy`
* `cartopy`
* `matplotlib`
* `netCDF4`
* `pupygrib`
* `joblib`
* `mpi4py`

## Single Location Scripts
* `script/london`
  * Training set: 1980-1989 inclusive
  * Test set: 1990-1999 inclusive
* `script/cardiff_20_20`
  * Training set: 1979-1999 inclusive
  * Test set: 2000-2019 inclusive

Run the script `hyper_slice.py` to run 10 000 MCMC samples. Afterwards, run `hyper_slice_forecast.py` to sample 1 000 forecast samples. Figures are plotted and saved in the `figure` directory.

The options may be provided which may be useful for development or debugging purposes. The following examples are provided:
* `python3 hyper_slice.py`
  * Does the default number of MCMC samples
  * If MCMC samples are detected from a previous run, only print out figures
* `python3 hyper_slice.py --sample 400`
  * Does 400 MCMC samples
* `python3 hyper_slice.py --sample 400` followed by `python3 hyper_slice.py --sample 800`
  * Does 400 MCMC samples, save the samples, then does 400 more MCMC samples
* `python3 hyper_slice_forecast.py`
  * Does the default number of forecast samples
  * If forecast samples are detected from a previous run, only print out figures
* `python3 hyper_slice_forecast.py --sample 400`
  * Does 400 forecast samples
* `python3 hyper_slice_forecast.py --sample 400` followed by `python3 hyper_slice_forecast.py --sample 800`
  * Does 400 forecast samples, save the samples, then does 400 more forecast samples
* `python3 hyper_slice_forecast.py --sample 400 --burnin 100`
  * Does 400 forecast samples with a burn in of 100. If `--burnin` is not provided, the default burn in is used.

Results are saved in the `result` directory. Delete it if you wish to restart the sampling process from the start.

## Multiple Locations Scripts
* `script/isle_of_man`
  * Training set: 1980-1989 inclusive
  * Test set: 1990-1999 inclusive
* `script/wales_10`
  * Training set: 1980-1989 inclusive
  * Test set: 1990-1999 inclusive
* `script/wales_10_20`
  * Training set: 1990-1999 inclusive
  * Test set: 2000-2019 inclusive
* `script/wales_20_20`
  * Training set: 1979-1999 inclusive
  * Test set: 2000-2019 inclusive

Run the script `downscale.py` and `dual.py` to do MCMC sampling without/with model field sampling respectively. Afterwards, run the script `downscale_forecast.py` and/or `dual_forecast.py` to do forecast sampling without/with model field sampling respectively.

The options may be provided which may be useful for development or debugging purposes. The following examples are provided:
* `python3 downscale.py`
  * Does the default number of MCMC samples
  * If MCMC samples are detected from a previous run, only print out figures
* `python3 downscale.py --sample 400`
  * Does 400 MCMC samples
* `python3 downscale.py --sample 400` followed by `python3 downscale.py --sample 800`
  * Does 400 MCMC samples, save the samples, then does 400 more MCMC samples
* `python3 downscale_forecast.py`
  * Does the default number of forecast samples
  * If forecast samples are detected from a previous run, only print out figures
* `python3 downscale_forecast.py --sample 400`
  * Does 400 forecast samples
* `python3 downscale_forecast.py --sample 400` followed by `python3 downscale_forecast.py --sample 800`
  * Does 400 forecast samples, save the samples, then does 400 more forecast samples
* `python3 downscale_forecast.py --sample 400 --burnin 100`
  * Does 400 forecast samples with a burn in of 100. If `--burnin` is not provided, the default burn in is used.
* `python3 downscale_forecast.py --noprint`
  * Does not print forecast figures

The code uses multiple threads so using a multi-core processor(s) is recommended.

## Notes for Developers
Please see the package `compound_poisson` for under the hood code and documentations.
