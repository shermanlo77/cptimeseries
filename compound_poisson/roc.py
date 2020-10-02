from matplotlib import pyplot as plt
import numpy as np

from compound_poisson import multiprocess

class Roc(object):

    def __init__(self, rain_warning, p_rain_warning, rain_true, pool=None):
        """
        Args:
            rain_warning: the amount of precipitation to detect
            p_rain_warning: forecasted probability of precipitation more than
                rain_warning, array, for each time point
            rain_true: actual observed precipitation, array, for each time point
        """
        self.rain_warning = rain_warning
        self.true_positive_array = None
        self.false_positive_array = None
        self.area_under_curve = None

        #get the times it rained more than rain_warning
        is_warning = rain_true > self.rain_warning
        #number of times the event happened
        n_is_warning = np.sum(is_warning)
        #number of times event did not happaned
        n_is_not_warning = len(is_warning) - n_is_warning

        #for each positive probability, sort them (highest to lowest) and they
            #will be used for thresholds. Highest to lowest so start with lowest
            #false positive, i.e. left to right on ROC curve
        threshold_array = p_rain_warning[is_warning]
        threshold_array = np.flip(np.sort(threshold_array))
        threshold_array = threshold_array[threshold_array > 0]

        #array to store true and false positives, used for plotting
        self.true_positive_array = [0.0]
        self.false_positive_array = [0.0]

        #for each threshold, get true and false positive
        message_array = []
        for threshold in threshold_array:
            message = PositiveRateMessage(p_rain_warning, is_warning, threshold)
            message_array.append(message)
        if pool is None:
            pool = multiprocess.Serial()
        pool_result = (
            pool.map(PositiveRateMessage.get_positive_rate, message_array))
        for true_positive, false_positive in pool_result:
            self.true_positive_array.append(true_positive)
            self.false_positive_array.append(false_positive)
        self.true_positive_array.append(1.0)
        self.false_positive_array.append(1.0)

        #calculate area under curve
        area_under_curve = []
        for i, true_positive in enumerate(self.true_positive_array):
            if i < len(self.true_positive_array)-1:
                height = (self.false_positive_array[i+1]
                    - self.false_positive_array[i])
                area_i = height * true_positive
                area_under_curve.append(area_i)
        self.area_under_curve = np.sum(area_under_curve)

    def plot(self):
        label = (str(self.rain_warning)+" mm, AUC = "
            +"{:0.3f}".format(self.area_under_curve))
        plt.step(self.false_positive_array,
                 self.true_positive_array,
                 where="post",
                 label=label)
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")

class PositiveRateMessage(object):

    def __init__(self, p_rain_warning, is_warning, threshold):
        self.p_rain_warning = p_rain_warning
        self.is_warning = is_warning
        self.threshold = threshold

    def get_positive_rate(self):
        n_is_warning = np.sum(self.is_warning)
        n_is_not_warning = len(self.is_warning) - n_is_warning
        positive = self.p_rain_warning >= self.threshold
        true_positive = (
            np.sum(np.logical_and(positive, self.is_warning)) / n_is_warning)
        false_positive = (
            np.sum(np.logical_and(positive, np.logical_not(self.is_warning)))
            / n_is_not_warning)
        return (true_positive, false_positive)
