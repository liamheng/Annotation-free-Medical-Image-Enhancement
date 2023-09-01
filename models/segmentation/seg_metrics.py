# -*- coding: UTF-8 -*-
"""
@Function:
@File: seg_metrics.py
@Date: 2022/1/15 11:04 
@Author: Hever
"""
import numpy as np


def _accuracy(tp, fp, tn, fn):
    # for numerical stability
    epsilon = 10e-16
    return (tp + tn) / (tp + fn + fp + tn + epsilon)


def _recall(tp, fn):
    # for numerical stability
    epsilon = 10e-16
    return tp / (tp + fn + epsilon)


def _dice(tp, fp, fn):
    # for numerical stability
    epsilon = 10e-16
    return 2 * tp / (2 * tp + fp + fn + epsilon)


def _precision(tp, fp):
    # for numerical stability
    epsilon = 10e-16
    return tp / (tp + fp + epsilon)


def _f1_score(recall, precision):
    # for numerical stability
    epsilon = 10e-16
    return 2 * (recall * precision) / (recall + precision + epsilon)


def _intersection_over_union(tp, fp, fn, tn):
    # for numerical stability
    epsilon = 10e-16
    union = tp + fp + fn
    if tp == 0 and fp == 0 and fn == 0 and tn > 0:
        """
        Special case: If there were only true negatives, set IoU to 1.
        This means the model correctly saw the absence of the true class.
        """
        return 1.0
    else:
        return tp / (union + epsilon)


def _kappa(tp, fp, tn, fn, n_pixel):
    # from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4372765/
    # for numerical stability
    epsilon = 10e-16
    p11 = tp / n_pixel
    p00 = tn / n_pixel
    p1plus = (tp + fp) / n_pixel
    pplus1 = (tp + fn) / n_pixel
    p0plus = (fn + tn) / n_pixel
    pplus0 = (fp + tn) / n_pixel

    return (p11 + p00 - (p1plus * pplus1 + p0plus * pplus0)) / (1 - (p1plus * pplus1 + p0plus * pplus0) + epsilon)


def tpfptnfn(y_true, y_pred, valid_mask):
    """Compute true positives (tp), false positives (fp), true negatives (tn) and false negatives (fn)
    Compute true positives (tp), false positives (fp), true negatives (tn) and false negatives (fn) for a given pair
    of reference masks and segmentation output. Take a valid mask into account, if one is provided. The valid mask
    indicates valid pixel locations with 1, invalid pixel locations with zero.
    :param y_true: ndarray: reference mask of the scene/tile(s)
    :param y_pred: ndarray: prediction output
    :param valid_mask: ndarray or None: valid pixel mask of the scene/tiles(s)
    :return dictionary of tp, fp, tn and fn
    """

    if valid_mask is None:
        # if no valid mask is provided,assume all pixels are valid
        valid_mask = np.ones(y_true.shape)
    return {
        "tp": np.sum(
            np.logical_and(
                np.logical_and([y_true == 1], [y_pred == 1]),
                [valid_mask == 1],
            )
        ),
        "fp": np.sum(
            np.logical_and(
                np.logical_and([y_true == 0], [y_pred == 1]),
                [valid_mask == 1],
            )
        ),
        "tn": np.sum(
            np.logical_and(
                np.logical_and([y_true == 0], [y_pred == 0]),
                [valid_mask == 1],
            )
        ),
        "fn": np.sum(
            np.logical_and(
                np.logical_and([y_true == 1], [y_pred == 0]),
                [valid_mask == 1],
            )
        ),
        "n_valid_pixel": np.count_nonzero(valid_mask),
    }


def segmentation_metrics(tpfptnfn, decimals=4):
    """Compute segmentation performance metrics.
    Compute the segmentation performance metrics given the true positives tp, false positives fp, true negatives tn,
    false negatives fn and number of valid pixels. Compute metrics: Accuracy, Recall, Precision, F1, IoU and Kappa.
    :param tpfptnfn: Dict. Contains tp, fp, tn, fn and n_valid_pixel {"tp": 100, ...}
    :param decimals: The number of decimals to use when rounding the number (default 4)
    :return dictionary containing metrics
    """
    tp = tpfptnfn["tp"]
    fp = tpfptnfn["fp"]
    tn = tpfptnfn["tn"]
    fn = tpfptnfn["fn"]
    n_pixel = tpfptnfn["n_valid_pixel"]

    acc = _accuracy(tp, fp, tn, fn)
    rec = _recall(tp, fn)
    prec = _precision(tp, fp)
    f1 = _f1_score(rec, prec)
    iou = _intersection_over_union(tp, fp, fn, tn)
    kap = _kappa(tp, fp, tn, fn, n_pixel)
    dice = _dice(tp, fp, fn)

    return {
        "iou": round(iou, decimals),
        "recall": round(rec, decimals),
        "precision": round(prec, decimals),
        "acc": round(acc, decimals),
        "F1": round(f1, decimals),
        "kappa": round(kap, decimals),
        "dice": round(dice, decimals)
    }