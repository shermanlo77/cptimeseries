from matplotlib import pyplot as plt
import numpy as np

def get_roc_curve(rain_warning, p_rain_warning, rain_true):
    """Return plots for ROC curve and area under the curve

    Args:
        rain_warning: the amount of precipitation to detect
        p_rain_warning: forecasted probability of precipitation more than
            rain_warning, array, for each time point
        rain_true: actual observed precipitation, array, for each time point

    Return:
        tuple, 3 elements, array of true positives, array of false positives,
            area under the curve. To plot the ROC curve, use
            matplotlib.pyplot.step() with where="post"
    """
    #for each positive probability, sort them (highest to lowest) and they will
        #be used for thresholds. Highest to lowest so start with lowest false
        #positive, i.e. left to right on ROC curve
    threshold_array = np.flip(np.sort(p_rain_warning))
    threshold_array = threshold_array[threshold_array > 0]

    #get the times it rained more than rain_warning
    is_warning = rain_true > rain_warning
    #number of times the event happened
    n_is_warning = np.sum(is_warning)
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

    #calculate area under curve
    area_under_curve = []
    for i, true_positive in enumerate(true_positive_array):
        if i < len(true_positive_array)-1:
            area_under_curve.append(
                (false_positive_array[i+1] - false_positive_array[i])
                * true_positive)
    area_under_curve = np.sum(area_under_curve)

    return (true_positive_array, false_positive_array, area_under_curve)

def plot_roc_curve(true_positive_array,
                   false_positive_array,
                   auc,
                   rain_warning):
    plt.step(false_positive_array,
             true_positive_array,
             where="post",
             label=str(rain_warning)+" mm, AUC = "+str(round(auc, 3)))
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
