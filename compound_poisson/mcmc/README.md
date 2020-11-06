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

- `mcmc_abstract.ReadOnlyFromSlice`
  - &#x25C7;-1 `mcmc_abstract.Mcmc`

The class `mcmc_abstract.ReadOnlyFromSlice` is a wrapper class which extract MCMC samples from a larger MCMC sampling scheme. For example, extract MCMC samples from one particular location from a multiple location sampling scheme.


- `time_series.TimeSeriesMcmc`
  - &#x25C7;-1..\*`mcmc_abstract.Mcmc`


- `downscale.Downscale`
  - &#x25C7;-1..\*`mcmc_abstract.Mcmc`

Instances of `time_series.TimeSeriesMcmc` and  `downscale.Downscale` will have a `mcmc_abstract.Mcmc` object for each component to sample. They are all sampled using randon scan Gibbs sampling.

## Target Distributions


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
