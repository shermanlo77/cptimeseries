"""Wrapper functions around a Fitter to do fitting and forecasting

Wrapper functions around a Fitter, making it suitable to run as a script. The
    package argparse is used to set user requested values when fitting and
    forecasting.

For time series, use time_series_fit() and time_series_forecast()
For multiple locations, use downscale_fit() and downscale_forecast()
For argparse parameters, see get_fit_parser() and get_forecast_parser()
"""

import argparse

def time_series_fit(fitter, data, seed):
    """Do a fit for a given Fitter (for time series)
    """
    n_sample, is_print, n_thread = get_fit_parser()
    fitter.fit(data, seed, n_sample, is_print=is_print)

def time_series_forecast(fitter, training, test, default_burn_in):
    """Do a forecast for a given fitted Fitter (for time series)
    """
    n_simulation, is_print, burn_in = get_forecast_parser()
    if burn_in is None:
        burn_in = default_burn_in
    fitter.forecast((training, test), n_simulation, burn_in)

def downscale_fit(fitter, data, seed, Pool):
    """Do a fit for a given Fitter (for multiple locations)
    """
    n_sample, is_print, n_thread = get_fit_parser()
    pool = None
    if n_thread is None:
        pool = Pool()
    else:
        pool = Pool(n_thread)
    fitter.fit(data, seed, n_sample, pool, is_print)
    pool.join()

def downscale_forecast(fitter, test, default_burn_in, Pool):
    """Do a forecast for a given fitted Fitter (for multiple locations)
    """
    n_simulation, is_print, burn_in, n_thread = get_forecast_parser()
    if burn_in is None:
        burn_in = default_burn_in
    pool = None
    if n_thread is None:
        pool = Pool()
    else:
        pool = Pool(n_thread)
    fitter.forecast(test, n_simulation, burn_in, pool, is_print)
    pool.join()

def get_fit_parser():
    """Extract argparse parameters provided by the user for fitting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", help="number of mcmc samples", type=int)
    parser.add_argument("--noprint",
                        help="do not print figures",
                        default=False,
                        action="store_true")
    parser.add_argument("--core", help="number of threads", type=int)
    n_sample = parser.parse_args().sample
    is_print = not parser.parse_args().noprint
    n_thread = parser.parse_args().core
    return (n_sample, is_print, n_thread)

def get_forecast_parser():
    """Extract argparse parameters provided by the user for forecasting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", help="number of simulations", type=int)
    parser.add_argument("--noprint",
                        help="do not print figures",
                        default=False,
                        action="store_true")
    parser.add_argument("--burnin", help="burn in", type=int)
    parser.add_argument("--core", help="number of threads", type=int)
    n_simulation = parser.parse_args().sample
    is_print = not parser.parse_args().noprint
    burn_in = parser.parse_args().burnin
    n_thread = parser.parse_args().core
    return (n_simulation, is_print, burn_in, n_thread)
