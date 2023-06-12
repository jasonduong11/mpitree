import numpy as np


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
    return np.unique(x, return_counts=True)[1] / len(x)


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
    try:
        X.astype(np.float64)
        return True
    except ValueError:
        return False


np.proba = proba
np.split_mask = split_mask
