# -*- coding: utf-8 -*-

"""
Copyright 2016 Randal S. Olson

This file is part of the TPOT library.

The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along
with the TPOT library. If not, see http://www.gnu.org/licenses/.

"""
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

# def balanced_accuracy_score(y_true, y_pred):
#     """Default scoring function: balanced accuracy

#     Balanced accuracy computes each class' accuracy on a per-class basis using a
#     one-vs-rest encoding, then computes an unweighted average of the class accuracies.

#     Parameters
#     ----------
#     y_true: numpy.ndarray {n_samples}
#         True class labels
#     y_pred: numpy.ndarray {n_samples}
#         Predicted class labels by the estimator

#     Returns
#     -------
#     fitness: float
#         Returns a float value indicating the `individual`'s balanced accuracy
#         0.5 is as good as chance, and 1.0 is perfect predictive accuracy
#     """
#     all_classes = list(set(np.append(y_true, y_pred)))
#     all_class_accuracies = []
#     for this_class in all_classes:
#         this_class_sensitivity = \
#             float(sum((y_pred == this_class) & (y_true == this_class))) /\
#             float(sum((y_true == this_class)))

#         this_class_specificity = \
#             float(sum((y_pred != this_class) & (y_true != this_class))) /\
#             float(sum((y_true != this_class)))

#         this_class_accuracy = (this_class_sensitivity + this_class_specificity) / 2.
#         all_class_accuracies.append(this_class_accuracy)

#     return np.mean(all_class_accuracies)

def f1_macro(y_true,y_pred):
    return f1_score(y_true,y_pred,average='macro')

# Sklearn 0.21 version
def balanced_accuracy_score(y_true, y_pred, sample_weight=None,
                            adjusted=False):
    """Compute the balanced accuracy
    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
    The best value is 1 and the worst value is 0 when ``adjusted=False``.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, and perfect performance scores 1.
    Returns
    -------
    balanced_accuracy : float
    See also
    --------
    recall_score, roc_auc_score
    Notes
    -----
    Some literature promotes alternative definitions of balanced accuracy. Our
    definition is equivalent to :func:`accuracy_score` with class-balanced
    sample weights, and shares desirable properties with the binary case.
    See the :ref:`User Guide <balanced_accuracy_score>`.
    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced accuracy and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern
           Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
           `Fundamentals of Machine Learning for Predictive Data Analytics:
           Algorithms, Worked Examples, and Case Studies
           <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.
    Examples
    --------
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy_score(y_true, y_pred)
    0.625
    """
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score
