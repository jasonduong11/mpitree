import numpy as np
import pandas as pd


def _to_pandas_dataframe(X, y, /):
    """Converts a numpy array to pandas dataframe."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=map(str, range(X.ndim)))
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    return X, y