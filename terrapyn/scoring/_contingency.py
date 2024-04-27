import typing as T

import numpy as np
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    """
    Calculate a confusion matrix (contingency table) for observations (truth) and a model (predictions),
    with associated statistics.

    Args:
        truth: Array of values for observations (truth).
        prediction: Array of values for a model (predictions).
        normalize: Normalize the confusion matrix over the truth (rows), prediction (columns) conditions,
        or all the population, one of 'true', 'pred', 'all', None. If None, confusion matrix will not be normalized.
        labels_to_include: List of labels in the truth/prediction arrays to include when calculating the confusion
        matrix, such that all other labels will be ignored. If `None` include all labels.

    Confusion Matrix Statistics. Available attributes are:

    ==========    ================================================================
    ACC           Accuracy
    F1score       F1 score
    FDR           False Discovery Rate
    FNR           False Negative Rate (Miss rate)
    FOR           False Omission Rate
    FPR           False Positive Rate (probability of false alarm)
    NPV           Negative Predictive Value
    PPV           Positive Predictive Value (Precision)
    Prevalence    Prevalence
    TNR           True Negative Rate (Specificity, Selectivity)
    TPR           True Positive Rate (Recall, Sensitivity, probability of detection)
    TP            True Positive
    FP            False Positive
    FN            False Negative
    TN            True Negative
    TS            Threat Score / Critical Success Index
    ==========    ================================================================

    Input values must be positive and can be counts, percentage or decimal
    percentage.

    :attr float,int true_positive: True positive value
    :attr float,int false_positive: False positive value
    :attr float,int false_negative: False negative value
    :attr float,int true_negative: True negative value
    :attr float ACC: Accuracy
    :attr float F1score: F1 score
    :attr float FDR: False Discovery Rate
    :attr float FNR: False Negative Rate (Miss rate)
    :attr float FOR: False Omission Rate
    :attr float FPR: False Positive Rate (Probability of false alarm)
    :attr float NPV: Negative Predictive Value
    :attr float PPV: Positive Predictive Value (Precision)
    :attr float Prevalence: Prevalence
    :attr float TNR: True Negative Rate (Specificity, Selectivity)
    :attr float TPR: True Positive Rate (Recall, Sensitivity,
    Probability of detection)
    :attr float TS: Threat Score / Critical Success Index
    """

    def __init__(
        self,
        truth: np.ndarray = None,
        prediction: np.ndarray = None,
        normalize: str = "all",
        labels_to_include: T.Iterable = None,
    ):
        self.cm = confusion_matrix(truth, prediction, normalize=normalize, labels=labels_to_include)
        self.TN, self.FP, self.FN, self.TP = self.cm.ravel()

    def __repr__(self):
        return f"ConfusionMatrix(TP={self.TP}, FP={self.FP}, " f"FN={self.FN}, TN={self.TN})"

    @property
    def PPV(self):
        """Positive Predictive Value (Precision)"""
        return self.TP / (self.TP + self.FP)

    @property
    def FDR(self):
        """False Discovery Rate"""
        return self.FP / (self.TP + self.FP)

    @property
    def FOR(self):
        """False Omission Rate"""
        return self.FN / (self.FN + self.TN)

    @property
    def NPV(self):
        """Negative Predictive Value"""
        return self.TN / (self.FN + self.TN)

    @property
    def TPR(self):
        """True Positive Rate (Recall, Sensitivity, Probability of detection)"""
        return self.TP / (self.TP + self.FN)

    @property
    def FNR(self):
        """False Negative Rate (Miss rate)"""
        return self.FN / (self.FN + self.TN)

    @property
    def FPR(self):
        """False Positive Rate (probability of false alarm)"""
        return self.FP / (self.FP + self.TN)

    @property
    def TNR(self):
        """True Negative Rate (Specificity, Selectivity)"""
        return self.TN / (self.FP + self.TN)

    @property
    def Prevalence(self):
        """Prevalence"""
        return (self.TP + self.FN) / (self.TP + self.FP + self.FN + self.TN)

    @property
    def ACC(self):
        """Accuracy"""
        return (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)

    @property
    def F1score(self):
        """F1 score"""
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    @property
    def TS(self):
        """Threat Score / Critical Success Index"""
        return self.TP / (self.TP + self.FN + self.FP)
