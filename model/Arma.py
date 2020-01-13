class Arma:
    """Evaluates ARMA terms
    
    Evaluates ARMA terms. To be owned by instances of Parameter so that their
        methods ar(index) and ma(index) can be called here. AR and MA terms at
        the start of the time series are evaluated to be zero.
    
    Attributes:
        parameter: a Parameter instance which owns this arma
        time_series: a TimeSeries instance which owns parameter
    """
    
    def __init__(self, parameter):
        self.parameter = parameter
        self.time_series = parameter._parent
    
    def ar(self, index):
        """AR term at a time step
        
        Returns the AR term at a given time step.
        """
        if index > 0:
            return self.parameter.ar(self.parameter[index-1])
        else:
            return 0
        
    def ma(self, index):
        """MA term at a time step
        
        Returns the MA term at a given time step.
        """
        if index > 0:
            time_series = self.time_series
            return self.parameter.ma(time_series.y_array[index-1],
                                     time_series.z_array[index-1],
                                     time_series.poisson_rate[index-1],
                                     time_series.gamma_mean[index-1],
                                     time_series.gamma_dispersion[index-1])
        else:
            return 0

class ArmaForecastNoMa(Arma):
    """Evaluates ARMA terms for forecasting
    
    Evaluates ARMA terms. To be owned by instances of Parameter so that their
        methods ar(index) and ma(index) can be called here. AR and MA terms at
        the start of the time series are evaluated using the past (fitted) time
        series. MA terms, otherwise, are evaluated to be 0.
    """
    def __init__(self, parameter):
        super().__init__(parameter)
    
    def ar(self, index):
        return self.parameter.ar(self.parameter[index-1])
    
    def ma(self, index):
        if index == 0:
            time_series = self.time_series.fitted_time_series
            n = time_series.n
            return self.parameter.ma(time_series.y_array[n-1],
                                     time_series.z_array[n-1],
                                     time_series.poisson_rate[n-1],
                                     time_series.gamma_mean[n-1],
                                     time_series.gamma_dispersion[n-1])
        else:
            return 0

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
        return self.parameter.ar(self.parameter[index-1])
    
    def ma(self, index):
        if index == 0:
            time_series = self.time_series.fitted_time_series
            index = time_series.n
        else:
            time_series = self.time_series
        return self.parameter.ma(time_series.y_array[index-1],
                                 time_series.z_array[index-1],
                                 time_series.poisson_rate[index-1],
                                 time_series.gamma_mean[index-1],
                                 time_series.gamma_dispersion[index-1])
