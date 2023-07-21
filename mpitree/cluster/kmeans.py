import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class KMeans(BaseEstimator):
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        X = check_array(X)
        return self

    def predict(self, X):
        check_is_fitted(self)