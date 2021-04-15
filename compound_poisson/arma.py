"""For evaluating ARMA terms using an emulation

For evaluating ARMA terms using an emulation. Two classes, Arma and
    ArmaForecast. Arma is for general use, where ARMA terms before time point
    zero are zero. ArmaForecast is used for forecasting and continuing on a
    time series, uses the previous time series when elevating time points
    before zero (relatively).

Arma <- ArmaForecast
"""

import numpy as np


class Arma(object):
    """Evaluates ARMA terms

    Evaluates ARMA terms. To be owned by instances of Parameter so that their
        methods ar(), ma(), d_reg_ar() and d_reg_ma() can be called here. AR
        and MA terms at the start of the time series are evaluated to be zero.

    Attributes:
        parameter: a Parameter instance which owns this arma
        time_series: a TimeSeries instance which owns parameter
        n_ar: number of autocorrelation terms
        n_ma: number of moving average terms
    """

    def __init__(self, parameter):
        self.parameter = parameter
        self.time_series = parameter.time_series
        self.n_ar = parameter.n_ar
        self.n_ma = parameter.n_ma

    def ar_term(self, index):
        """AR term at a time step

        Returns the AR term (vector) at a given time step. Each element
            correspond to each lag [lag 1, lag 2, ... lag n_ar]

        Args:
            index: time step (t)

        Returns:
            vector of length self.n_ar
        """
        ar = np.zeros(self.n_ar)
        for i in range(self.n_ar):
            index_lag = index-i-1  # lag term, eg one step behind for i=0
            if index_lag >= 0:
                ar[i] = self.parameter.ar(index_lag)
            else:
                ar[i] = 0
        return ar

    def ma_term(self, index):
        """MA term at a time step

        Returns the MA term (vector) at a given time step. Each element
            correspond to each lag [lag 1, lag 2, ... lag n_ma]

        Args:
            index: time step (t)

        Returns:
            vector of length self.n_ma
        """
        time_series = self.time_series
        ma = np.zeros(self.n_ma)
        for i in range(self.n_ma):
            index_lag = index-i-1  # lag term, eg one step behind for i=0
            if index_lag >= 0:
                ma[i] = self.parameter.ma(
                    time_series[index_lag],
                    time_series.z_array[index_lag],
                    time_series.poisson_rate[index_lag],
                    time_series.gamma_mean[index_lag],
                    time_series.gamma_dispersion[index_lag])
            else:
                ma[i] = 0
        return ma

    def d_reg_ar_term(self, index, key):
        """Derivative of the AR term with respect to a regression parameter

        Return the derivative of the AR term \\sum_{i=1}^p\\phi_{i}\\Phi(t-i)
            with respect to a regression parameter (eg \\beta, \\phi, \\theta,
            k). The return value is a vector of the same shape as the
            regression parameter

        Args:
            index: time step (t)
            key: name of the regression parameter (eg "reg", "AR", "MA",
                "const")
        """
        grad = []  # array of gradient vectors, one for each AR lag
        parameter = self.parameter
        # for each AR lag
        for i in range(self.n_ar):
            index_lag = index - i - 1  # lag term, eg one step behind for i=0
            # work out gradient if there is a term
            if index_lag >= 0:
                grad.append(
                    parameter["AR"][i] * parameter.d_reg_ar(index_lag, key))
            # zero gradient if there is no term
            else:
                grad.append(np.zeros_like(parameter[key]))
        # sum gradient over all AR lags
        grad = np.asarray(grad)
        grad = np.sum(grad, 0)
        return grad

    def d_reg_ma_term(self, index, key):
        """Derivative of the MA term with respect to a regression parameter

        Return the derivative of the MA term
            \\sum_{i=1}^q\\theta_{i}\\Theta(t-i) with respect to a regression
            parameter (eg \\beta, \\phi, \\theta, k). The return value is a
            vector of the same shape as the regression parameter

        Args:
            index: time step (t)
            key: name of the regression parameter (eg "reg", "AR", "MA",
                "const")
        """
        grad = []  # array of gradient vectors, one for each AR lag
        parameter = self.parameter
        # for each MA lag
        for i in range(self.n_ma):
            index_lag = index - i - 1  # lag term, eg one step behind for i=0
            # work out gradient if there is a term
            if index_lag >= 0:
                grad.append(
                    parameter["MA"][i] * parameter.d_reg_ma(index_lag, key))
            # zero gradient if there is no term
            else:
                grad.append(np.zeros_like(parameter[key]))
        grad = np.asarray(grad)
        grad = np.sum(grad, 0)
        return grad


class ArmaForecast(Arma):
    """Evaluates ARMA terms for forecasting

    Evaluates ARMA terms. To be owned by instances of Parameter so that their
        methods ar(index) and ma(index) can be called here. AR and MA terms at
        the start of the time series are evaluated using the past (fitted) time
        series.
    """

    def __init__(self, parameter):
        super().__init__(parameter)

    def ar(self, index):
        ar = np.zeros(self.n_ar)
        for i in range(self.n_ar):
            ar[i] = self.parameter.ar(index-i-1)
        return ar

    def ma(self, index):
        time_series = self.time_series
        ma = np.zeros(self.n_ma)
        for i in range(self.n_ma):
            index_lag = index-i-1
            if index_lag > 0:
                time_series = self.time_series
            else:
                time_series = self.time_series.fitted_time_series
                index_lag = len(time_series)-i-1
            ma[i] = self.parameter.ma(
                time_series[index_lag],
                time_series.z_array[index_lag],
                time_series.poisson_rate[index_lag],
                time_series.gamma_mean[index_lag],
                time_series.gamma_dispersion[index_lag])
        return ma
