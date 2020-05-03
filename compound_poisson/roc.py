import numpy as np

def get_roc_curve(rain_warning, p_rain_warning, rain_true):
    """Plot ROC curve
    """
    #for each positive probability, sort them and they will be used for
        #thresholds
    threshold_array = np.flip(np.sort(p_rain_warning))
    threshold_array = threshold_array[threshold_array > 0]

    #get the times it rained more than rain_warning
    is_warning = rain_true > rain_warning
    n_is_warning = np.sum(is_warning) #number of times the event happened
    #number of times event did not happaned
    n_is_not_warning = len(is_warning) - n_is_warning
    #array to store true and false positives, used for plotting
    true_positive_array = [0.0]
    false_positive_array = [0.0]
    #for each threshold, get true and false positive
    for threshold in threshold_array:
        positive = p_rain_warning >= threshold
        true_positive = (
            np.sum(np.logical_and(positive, is_warning)) / n_is_warning)
        false_positive = (
            np.sum(np.logical_and(positive, np.logical_not(is_warning)))
            / n_is_not_warning)
        true_positive_array.append(true_positive)
        false_positive_array.append(false_positive)
    true_positive_array.append(1.0)
    false_positive_array.append(1.0)

    area_under_curve = []
    for i, true_positive in enumerate(true_positive_array):
        if i < len(true_positive_array) -1:
            area_under_curve.append(
                (false_positive_array[i+1] - false_positive_array[i])
                * true_positive)
    area_under_curve = np.sum(area_under_curve)

    return (true_positive_array, false_positive_array, area_under_curve)
