"""For plotting the root mean squared bias vs root variance, ideally the curve
    should be on the diagonal line

BiasVarAnalyser
    <- TimeSeries
    <- Downscale
"""

from matplotlib import pyplot as plt
import numpy as np

N_BIN = 16

class BiasVarAnalyser(object):
    """Analyse the bias and variance relationship

    Base class for plotting the root mean squared bias vs root variance. Each
        rainfall in a point in space and time has a variance and bias. They are
        all pooled together and binned according to their variance. They are
        plotted by taking the root mean (ie *root mean* squared bias and *root*
        variance) for each bin.

    Attribues:
        forecaster: a Forecaster object
        binned_variance: array, variance for each bin
        binned_square_bias: array, squared bias for each bin
    """

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.binned_variance = []
        self.binned_square_bias = []

    def analyse(self):
        """Work out and save the member variables binned_variance and
            binned_square_bias
        """
        square_bias_array = self.get_square_bias()
        variance_array = self.get_variance()
        n_data = len(square_bias_array) #number of points in space and time

        #define the boundary for each bin, attempt for them to be of the same
            #size
        bin_index_boundary = np.round(np.linspace(0, n_data, N_BIN+1))
        bin_index_boundary = np.asarray(bin_index_boundary, dtype=np.int32)

        #sort the variance and bin the squared_bias and variance terms in order
            #of increasing variance
        var_sort_index = np.argsort(variance_array)
        for i_bin in range(N_BIN):
            #extract index for this bin
            index_0 = bin_index_boundary[i_bin]
            index_1 = bin_index_boundary[i_bin+1]
            index_array = var_sort_index[index_0: index_1]

            #get the data for this bin
            variance_i = variance_array[index_array]
            square_bias_i = square_bias_array[index_array]

            #take mean over this bin
            self.binned_variance.append(np.mean(variance_i))
            self.binned_square_bias.append(np.mean(square_bias_i))

    def plot(self, path_to_figure, cycler=None):
        """Plot root mean squared bias vs root variance
        """
        plt.figure()
        ax = plt.gca()
        if not cycler is None:
            ax.set_prop_cycle(cycler)
        plt.plot(
            np.sqrt(self.binned_variance), np.sqrt(self.binned_square_bias),
            '.-')

        max_axis = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])
        plt.plot([0, max_axis], [0, max_axis], 'k--')
        plt.xlabel("RMS spread (mm)")
        plt.ylabel("RMS error (mm)")
        plt.savefig(path_to_figure, bbox_inches="tight")
        plt.close()

    def get_variance(self):
        """Return array of variance, one element for each point in space and
            time
        """
        raise NotImplementedError

    def get_square_bias(self):
        """Return array of square bias, one element for each point in space and
            time
        """
        raise NotImplementedError

class TimeSeries(BiasVarAnalyser):

    def __init__(self, forecaster, observed):
        super().__init__(forecaster)
        self.observed = observed

    def get_variance(self):
        variance_array = []
        #loop over each sample
        for forecast_i in self.forecaster.forecast_array:
            variance_array.append(
                np.square(forecast_i - self.forecaster.forecast_median))
        variance_array = np.asarray(variance_array)
        #take mean over samples
        variance_array = np.mean(variance_array, axis=0)
        return variance_array

    def get_square_bias(self):
        return np.square(self.forecaster.forecast_median - self.observed)

class Downscale(BiasVarAnalyser):

    def __init__(self, forecaster):
        super().__init__(forecaster)

    def get_variance(self):
        variance_array = np.asarray([])
        forecaster = self.forecaster
        observed = self.forecaster.data
        #loop over locations
        for forecaster_i, observed_rain_i in (
            zip(forecaster.generate_time_series_forecaster(),
                observed.generate_unmask_rain())):

            #work out square difference for each point in space and time
            variance_i = (forecaster_i.forecast_array
                - np.tile(observed_rain_i, [forecaster.n_simulation, 1]))
            #take mean over simulation
            variance_i = np.mean(np.square(variance_i), axis=0)
            variance_array = np.concatenate((variance_array, variance_i))

            forecaster_i.del_memmap()

        return variance_array

    def get_square_bias(self):
        square_bias_array = np.asarray([])
        forecaster = self.forecaster
        observed = self.forecaster.data
        #loop over locations
        for forecaster_i, observed_rain_i in (
            zip(forecaster.generate_time_series_forecaster(),
                observed.generate_unmask_rain())):

            square_bias_i = np.square(
                forecaster_i.forecast_median - observed_rain_i)
            square_bias_array = np.concatenate(
                (square_bias_array, square_bias_i))

            forecaster_i.del_memmap()

        return square_bias_array
