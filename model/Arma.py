import numpy as np

class Arma:
    """Evaluates ARMA terms
    
    Evaluates ARMA terms. To be owned by instances of Parameter so that their
        methods ar(index) and ma(index) can be called here. AR and MA terms at
        the start of the time series are evaluated to be zero.
    
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
    
    def ar(self, index):
        """AR term at a time step
        
        Returns the AR term at a given time step.
        """
        ar = np.zeros(self.n_ar)
        for i in range(self.n_ar):
            index_lag = index-i-1
            if index_lag >= 0:
                ar[i] = self.parameter.ar(index_lag)
            else:
                ar[i] = 0
        return ar
    
    def ma(self, index):
        """MA term at a time step
        
        Returns the MA term at a given time step.
        """
        time_series = self.time_series
        ma = np.zeros(self.n_ma)
        for i in range(self.n_ma):
            index_lag = index-i-1
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
