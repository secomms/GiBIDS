# Functions for binary classification performance evaluation

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns


def get_other_metrics(y_true: pd.Series, y_pred: pd.Series):
    """
    Compute False Positive Rate, False Negative Rate and Area under the ROC curve
    :param y_true: Actual binary classes
    :param y_pred: Binary classes predicted by the tested classifier
    :return: computed metrics values
    """
    cm = confusion_matrix(y_true, y_pred)
    [[TN, FP], [FN, TP]] = cm
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    AUC = roc_auc_score(y_true, y_pred)
    return FPR, FNR, AUC


def get_confusion_matrix(y_true: pd.Series, y_pred: pd.Series):
    """
    Get Confusion Matrix in pandas.Dataframe() format and array format

    :param y_true: Actual binary classes
    :param y_pred: Binary classes predicted by the tested classifier
    :return: Array
    """
    labs = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labs)  # .ravel()
    cm_df = pd.DataFrame(cm, index=labs, columns=labs)
    plt.figure(figsize=(3, 3))
    sns.heatmap(cm_df, annot=True, fmt='g')  # font size
    plt.show()
    return cm_df, cm


def get_metrics(y_true: pd.Series, y_pred: pd.Series):
    """
    Create a custom Classification Report for Binary Classification results containing listed metrics for each class:
        --> Accuracy
        --> Precision
        --> Recall
        --> F1-Score
        --> False Positive Rate (FPR)

    :param y_true: Actual binary classes
    :param y_pred: Binary classes predicted by the tested classifier
    :return: Confusion Matrix image printed and pandas.Dataframe() containing all results
    """
    cm_df, cm = get_confusion_matrix(y_true, y_pred)
    FP = cm_df.sum(axis=0) - np.diag(cm_df)
    FN = cm_df.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm_df)
    TN = cm_df.values.sum() - (FP + FN + TP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    TPR = TP / (TP + FN)  # RECALL
    PPV = TP / (TP + FP)  # PRECISION
    F1 = (2 * TPR * PPV) / (TPR + PPV)
    # FPR = FP / (FP + TN)  # FALSE ALARM RATE
    results = pd.DataFrame()
    results['Accuracy'] = round(ACC, 4)
    results['Precision'] = round(PPV, 4)
    results['Recall'] = round(TPR, 4)
    results['F1-Score'] = round(F1, 4)
    # results['FPR'] = round(FPR, 4)
    results['Support'] = TP + FN
    results = results.fillna('-')
    return results


def get_classification_errors(y_pred: pd.Series, test_df: pd.DataFrame, binary_label: str, multiclass_label: str):
    """
    Map for the test set the errors made in binary prediction in multi-class errors

    :param y_pred: Binary classes predicted by the tested classifier
    :param test_df: pandas.Dataframe() containing the test set (features + binary label + multi-class label)
    :param binary_label: column name of binary label
    :param multiclass_label: column name of multi-class label
    :return: pandas.Dataframe() with class name, number of errors and error percent for each class (multi)
    """
    y_true = test_df[binary_label].values
    classes = test_df[multiclass_label].unique()
    errors = []
    percents = []
    for cls in classes:
        idx = test_df[multiclass_label] == cls
        y_true_cls = y_true[idx]
        y_pred_cls = y_pred[idx]
        cm_cls = confusion_matrix(y_true_cls, y_pred_cls)
        errors_cls = sum(sum(cm_cls)) - sum(cm_cls.diagonal())
        errors.append(errors_cls)
        percents.append(errors_cls / len(y_true_cls))
    df = pd.DataFrame({'errors': errors, 'error_percent': percents}, index=classes)
    df.sort_index(ascending=True, inplace=True)
    return df
