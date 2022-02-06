"""
Metrics to implement from DeepSonar: Accuracy, AUC, F1, AP, FPR, FNR, EER.
I only have EER at the moment.
"""

def calculate_eer(truths, predictions):
    """ 
    Calculates the EER between a set of truth values and predicted values.
    The lower the EER, the better the performance of the model.
    This was implemented by the authors of the VGG model.
    Same function is present in evaluate_vgg_model.py, but this one is used for DeepSonar.

    More info on calculating EER:
    stack overflow: https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
    other link: https://yangcha.github.io/EER-ROC/

    Arguments:
        truths: the truth values
        predictions: the predicted values
    Returns:
        the EER and some other value "thresh" - don't understand this right now.
    """
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(truths, predictions, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh