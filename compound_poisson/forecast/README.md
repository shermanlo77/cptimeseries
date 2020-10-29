# compound_poisson.forecast

For forecasting and assessing the performance of the forecast.

## Forecasting

- `compound_poisson.forecast.forecast_abstract.Forecaster`
  - &larr; `compound_poisson.forecast.time_series.Forecaster`
    - &larr; `compound_poisson.forecast.time_series.SelfForecaster`
    - &larr; `compound_poisson.forecast.downscale.TimeSeriesForecaster`
  - &larr; `compound_poisson.forecast.downscale.Forecaster`


- `compound_poisson.time_series.TimeSeries`
  - &#x25C7;-1 `compound_poisson.forecast.time_series.Forecaster`
  - &#x25C7;-1 `compound_poisson.forecast.time_series.SelfForecaster`


- `compound_poisson.downscale.Downscale`
  - &#x25C7;-1  `compound_poisson.forecast.downscale.TimeSeriesForecaster`
  - &#x25C7;-1..\* `compound_poisson.downscale.TimeSeriesDownscale`
    - &#x25C7;-1 `compound_poisson.forecast.downscale.TimeSeriesForecaster`

The abstract superclass is `compound_poisson.forecast.forecast_abstract.Forecaster`.
- Implementations for a single location time series are `time_series.Forecaster`, for forecasting the test set, and `compound_poisson.forecast.time_series.SelfForecaster`, for forecasting the training set.
- For multiple locations, aka downscale, implementations are `compound_poisson.forecast.downscale.Forecaster`, which handles the forecast for multiple locations, and `compound_poisson.forecast.downscale.TimeSeriesForecaster` which handles the forecast for each location.

Instances from these classes can do forward simulation by calling the appropriate methods in `compound_poisson.time_series.TimeSeries` or `compound_poisson.downscale.Downscale`. Ensemble of these forecasts are saved as `numpy.memmap` objects so that they are saved onto a drive and not use too much RAM.

For forecasting the test set, an instance from the classes `compound_poisson.forecast.time_series.Forecaster` and `compound_poisson.forecast.downscale.Forecaster`are owned and used by `compound_poisson.time_series.TimeSeries` or `compound_poisson.downscale.Downscale` respectively.

For multiple locations, multiple instances of `compound_poisson.downscale.TimeSeriesDownscale` are owned by `compound_poisson.downscale.Downscale`. Each instance of `compound_poisson.downscale.TimeSeriesDownscale` owns an instance of `compound_poisson.forecast.downscale.TimeSeriesForecaster`. The design was such that each location has a `compound_poisson.forecast.downscale.TimeSeriesForecaster` object. An instance of `compound_poisson.forecast.downscale.Forecaster` will modify each location's `compound_poisson.forecast.downscale.TimeSeriesForecaster` object to do forecasting. The reason `compound_poisson.forecast.downscale.TimeSeriesForecaster` was implemented was because all locations can share the same memmap file, writing and reading to it. Various methods are overridden to achieve this behaviour.

Direct access to instances of these classes should only be needed when accessing and assessing forecasts. Methods in `compound_poisson.time_series.TimeSeries` and `compound_poisson.downscale.Downscale` should be available to encapsulate the forecasting using these instances.

## Assessing the Forecast

Comparing the forecast with the observed. All figures are produced in the module `compound_poisson.forecast.print` which uses the following modules:
- `compound_poisson.forecast.coverage_analysis` counts the number of days from the daily forecasts which are captured by a specified credible interval and records the width of that interval,
- `compound_poisson.forecast.distribution_compare` compares the empirical distribution of the forecast with the observed,
- `compound_poisson.forecast.loss` compares the forecast with the observed using loss functions,
- `compound_poisson.forecast.residual_analysis` investigates any trends with the residuals using graphs,
- `compound_poisson.forecast.roc` plot ROC curves,
- `compound_poisson.forecast.time_segmentation` splits the forecast into segments such as yearly segments or for every season,
- `compound_poisson.forecast.loss_segmentation` obtains the loss function for specified segments of the forecast.
