""""""

import numpy as np
from sklearn.utils.validation import check_array


def proba(x):
    """Short Summary

    Extended Summary

    Parameters
    ----------
    x : np.ndarray, ndim=1

    Returns
    -------
    np.ndarray
    """
    _, n_class_dist = np.unique(x, return_counts=True)
    return n_class_dist / len(x)


def split_mask(X, mask):
    """Short Summary

    Extended Summary

    Parameters
    ----------
    X : np.ndarray, ndim=2
    mask : bool

    Returns
    -------
    list
    """
    return [X[mask], X[~mask]]


def is_numeric_dtype(X):
    """Short Summary

    Extended Summary

    Parameters
    ----------
    X : np.ndarray

    Returns
    -------
    bool
    """
    X = check_array(X, ensure_2d=False, dtype=object)

    try:
        X.astype(np.float64)
        return True
    except ValueError:
        return False


np.proba = proba
np.split_mask = split_mask
