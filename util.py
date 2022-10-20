import numpy as np
import math
from sklearn.metrics import f1_score


def h_score_loss(x_, alpha=0.5):
    """
    Calculates the heterogeneity score (H-score) loss of the array of feature importance x_
    Parameters
    ----------
    x_: array of feature importance
    alpha: coefficient (0 < alpha < 1) that defines the weight that features with value less than the mean have.
        With alpha=0.5 all features have the same weight, regardless if their value is below or above the mean.
        A value of alpha below 0.5 gives more weight to features with value above the mean.
    Returns
    -------
    result : H-score loss for the array of feature importance x_ in range [0, 1]
    """
    if not alpha > 0 and alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (not included)")
    n = len(x_)
    mean = np.sum(x_)/n
    s = 0
    for x in x_:
        f = (1-alpha) * (x-mean) if x > 0 else alpha * (x-mean)
        s += np.abs(f)  # f * f
    loss = s/n
    std_loss = (1 / (1 + math.exp(-loss)) - 0.5) * 2  # apply shifted sigmoid function to put the loss in range [0,1]
    # score = 1 / std_loss if std_loss != 0 else 1000  # ensure we don't divide by 0; if loss = 0 -> very high score
    return std_loss


def custom_score(y_true, y_pred, feature_importance, verbose=False, het_weight=0.5):
    """
    The custom score which considers both the F1-Score and the heterogeneity metrics
    :param y_true: ground truth labels
    :param y_pred: predicted labels
    :param feature_importance: array containing the importance for each feature
        :param verbose: whether to print the f1-score and the h-score in the console
    :param het_weight: the weight to assign to the heterogeneity in the loss
    :return:
    """
    f1 = f1_score(y_true, y_pred)
    het_loss = h_score_loss(feature_importance, alpha=0.5)
    score = f1 - het_loss*het_weight
    if verbose:
        print(f"Score: {score}. F1: {f1}, H: {het_loss}")
    return score
