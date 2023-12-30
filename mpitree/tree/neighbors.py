import numpy as np

from scipy.stats import mode
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._node import Node
from .decision_tree import BaseDecisionTree  # refactor tree class -> _node.py


class KDTree(BaseDecisionTree):
    def __init__(self, *, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : array-like
            2D feature array with shape (n_samples, n_features) of
            numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        Raises
        ------
        ValueError
            If `n_neighbors` attribute is not an integer type
        """
        X, y = check_X_y(X, y)

        if not isinstance(self.n_neighbors, int):
            raise ValueError("`n_neighbors` attribute must be an integer type.")

        self.n_features_ = X.shape[1]
        self.tree_ = self._make_tree(X, y)

        return self

    def _make_tree(self, X, y, *, parent=None):
        if len(X) <= self.n_neighbors:
            # hold the idx of samples
            return Node(value=y, parent=parent)

        split_node = ...
        region_mask = ...

        split_node.left = self._make_tree(
            X[region_mask],
            y[region_mask],
            parent=split_node,
        )

        split_node.right = self._make_tree(
            X[~region_mask],
            y[~region_mask],
            parent=split_node,
        )

        return split_node

    def predict(self, X):
        """Return the predicated leaf decision node.

        Extended Summary

        Parameters
        ----------
        X : array-like
            2D test feature array with shape (n_samples, n_features) of
            numerical values.

        Returns
        -------
        ndarray
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        def compute_majority_vote(x):
            neighbors = mode(x, keepdims=False)
            return neighbors.mode

        return np.vectorize(compute_majority_vote)(self._decision_paths(X))

    def score(self, X, y):
        """Evaluate the performance of an approximate neighbors classifier.

        The metric used to evaluate the performance of an approximate neighbors
        classifier is `sklearn.metrics.accuracy_score`.

        Parameters
        ----------
        X : array-like
            2D test feature array with shape (n_samples, n_features) of numerical values.

        y : array-like
            1D test target array with shape (n_samples,) of categorical values.

        Returns
        -------
        float
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y, dtype=object)

        y_pred = self.predict(X)
        return np.sum(y, y_pred) / len(y)
