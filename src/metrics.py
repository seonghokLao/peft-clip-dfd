import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics as M


def ovr_roc(labels: np.ndarray, probs: np.ndarray):
    """
    Calculate the One-vs-Rest (OvR) Receiver Operating Characteristic (ROC) and Area Under the ROC Curve (AUROC) for each class.

    Parameters:
    labels (np.ndarray): Array of true class labels.
    probs (np.ndarray): Array of predicted probabilities for each class.

    Returns:
    tuple: A tuple containing:
        - aurocs (list): List of AUROC values for each class.
        - fprs (list): List of false positive rates for each class.
        - tprs (list): List of true positive rates for each class.
        - ths (list): List of thresholds for each class.
        - ovr_macro_auroc (float): Macro-averaged AUROC for the OvR setting.
    """
    num_classes = probs.shape[1]
    labels_one_hot = np.eye(num_classes)[labels]
    fprs, tprs, ths = [], [], []

    # Why OvR with macro avg: https://chatgpt.com/share/677e448d-5bc0-8006-b9b5-081427b02857
    ovr_macro_auroc = M.roc_auc_score(labels_one_hot, probs, multi_class="ovr", average="macro")

    # Calculate OvR ROC and AUROC for each class
    for i in range(num_classes):
        _fpr, _tpr, _ths = M.roc_curve(labels_one_hot[:, i], probs[:, i])
        fprs.append(_fpr)
        tprs.append(_tpr)
        ths.append(_ths)

    return fprs, tprs, ths, ovr_macro_auroc


def ovr_prc(labels: np.ndarray, probs: np.ndarray):
    """
    Calculate the One-vs-Rest (OvR) Precision-Recall Curve (PRC) and the mean Average Precision (mAP) for a multi-class classification problem.

    Args:
        labels (np.ndarray): Array of true class labels with shape (n_samples,).
        probs (np.ndarray): Array of predicted probabilities with shape (n_samples, n_classes).

    Returns:
        tuple: A tuple containing:
            - precs (list of np.ndarray): List of precision values for each class.
            - recs (list of np.ndarray): List of recall values for each class.
            - ths (list of np.ndarray): List of threshold values for each class.
            - ovr_macro_ap (float): The mean Average Precision (mAP) score.
    """
    num_classes = probs.shape[1]
    labels_one_hot = np.eye(num_classes)[labels]
    precs, recs, ths = [], [], []

    # The same as mAP (mean Average Precision)
    ovr_macro_ap = M.average_precision_score(labels_one_hot, probs, average="macro")

    # Calculate OvR PRC for each class
    for i in range(num_classes):
        _prec, _rec, _ths = M.precision_recall_curve(labels_one_hot[:, i], probs[:, i])
        precs.append(_prec)
        recs.append(_rec)
        ths.append(_ths)

    return precs, recs, ths, ovr_macro_ap


# TODO: verify claim
#! Notice: this might not work as expected if p(y=1) != 1 - p(y=0)
def calculate_eer(y_true: np.ndarray, y_score: np.ndarray):
    """
    Returns the equal error rate for a binary classifier output
    """
    fpr, tpr, thresholds = M.roc_curve(y_true, y_score[:, 1], pos_label=1)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except ValueError:
        eer = 1.0
    return eer
