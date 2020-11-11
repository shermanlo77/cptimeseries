# compound_poisson.mcmc

Modules for Bayesian inference using Markov chains Monte Carlo (MCMC). Can be summarised into two parts:
  - Modules for conducting MCMC algorithms `mcmc_*`,
  - Modules for evaluating the posterior distribution, also known as the target distribution `target*`.

## MCMC

- `mcmc_abstract.Mcmc`
  - &larr; `mcmc_abstract.ReadOnlyFromSlice`
  - &larr; `mcmc_parameter.Rwmh`
  - &larr; `mcmc_parameter.Elliptical`
  - &larr; `mcmc_z.ZRwmh`
  - &larr; `mcmc_z.ZSlice`
  - &larr; `mcmc_z.ZMcmcArray`

The base abstract class is `mcmc_abstract.Mcmc`. MCMC targetting continuous distributions are in `mcmc_parameter` and for discrete distributions are in
`mcmc_z`. The following MCMC algorithms were implemented:
- Random walk Metropolis-Hastings in `mcmc_z.ZRwmh`,
- Adaptive random walk Metropolis-Hastings in `mcmc_parameter.Rwmh`,
- Slice sampling in `mcmc_z.ZSlice`,
- Multiple slice sampling in `mcmc_z.ZMcmcArray`,
- Elliptical slice sampling in `mcmc_parameter.Elliptical`,
- \* within Gibbs sampling  in `mcmc_abstract.do_gibbs_sampling`.

References to these MCMC algorithms are included at the bottom.

## Target Distributions
- `target.Target`
  - `target.time_series.TargetParameter`
  - `target.time_series.TargetZ`
  - `target.time_series.TargetPrecision`

The base abstract class is `target.Target`. For `TimeSeries`, the implementations are `target.time_series.TargetParameter`, `TargetZ` and `TargetPrecision`. They represent the posterior distribution (up to a constant) of the regression parameters beta, latent variables z and hyper parameters tau (tuning or smoothing parameters in frequentist literature) respectively.

- `target.Target`
  - `target.downscale.TargetParameter`
  - `target.downscale.TargetGp`

For `Downscale`, the implementations are `target.downscale.TargetParameter` and `TargetGp`. They represent the posterior distribution (up to a constant) of the regression parameters beta, and hyper parameters tau (tuning or smoothing parameters in frequentist literature) respectively.

Default priors and hyper parameters are in `target`. See functions:
  - `get_parameter_mean_prior()`
  - `get_parameter_std_prior()`
  - `get_precision_prior()`
  - `get_gp_precision_prior()`

## Class Structure

- `time_series.TimeSeriesMcmc`
  - &#x25C7;-1..\*`mcmc_abstract.Mcmc`


- `downscale.Downscale`
  - &#x25C7;-1..\*`mcmc_abstract.Mcmc`

The intention was that instances of `time_series.TimeSeriesMcmc` and  `Downscale` will have a `mcmc_abstract.Mcmc` object for each component to sample. They are all sampled using randon scan Gibbs sampling.

- `mcmc_abstract.Mcmc`
  - &#x25C7;1- `target.Target`

Each `mcmc_abstract.Mcmc` has a `target.Target` distribution to sample.

However, because of the Bayesian hierarchical structure, some `target.Target` instances will want to modify other `target.Target` instances, each owned by `mcmc_abstract.Mcmc`. To tackle this, `time_series.TimeSeriesMcmc` and `downscale.Downscale` instances will have pointers to all `target.Target` instances. Some implementations of `mcmc_abstract.Mcmc` may contain a pointer to the parent instance of `time_series.TimeSeriesMcmc` or `downscale.Downscale` so that `target.Target` instances can communicate to each other when required.


## References

- Metropolis-Hastings
  - Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., and Teller, E. (1953). Equation of state calculations by fast computing machines. *The Journal of Chemical Physics*, 21(6):1087–1092.
  - Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1):97–109.
- Adaptive Metropolis-Hastings
  - Haario, H., Saksman, E., and Tamminen, J. (2001).  An adaptive Metropolis algorithm. *Bernoulli*, 7(2):223–242.
  - Roberts, G. O. and Rosenthal, J. S. (2009). Examples of adaptive MCMC. *Journal of Computational and Graphical Statistics*, 18(2):349–367.
- Slice sampling
  - Neal, R. M. (2003). Slice sampling. *The Annals of Statistics*, 31(3):705–741.
- Elliptical slice sampling
  - Murray, I., Adams, R. P., and MacKay, D. J. (2010). Elliptical slice sampling. *In Proceedings of the 13th International Conference on Artificial Intelligence and Statistics*.
- Gibbs sampling
  - Geman, S. and Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. *Institute of Electrical and Electronics Engineers Transactions on Pattern Analysis and Machine Intelligence*, PAMI-6(6):721–741.
