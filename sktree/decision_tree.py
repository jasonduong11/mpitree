from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
from statistics import mode

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._node import DecisionNode


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *, criterion=None):
        super().__init__()
        self.criterion_ = criterion

    def __iter__(self, other: DecisionNode = None):
        """


        Parameter
        --------

        Returns
        -------
        """
        check_is_fitted(self)

        if other is None:
            other = self.tree_

        yield other
        for _, child_node in sorted(other.children.items(), key=lambda u: u[1].is_leaf):
            yield from self.__iter__(child_node)

    def __repr__(self):
        """


        Parameter
        --------

        Returns
        -------
        """
        check_is_fitted(self)
        return "\n".join(map(str, self))

    def fit(self, X, y):
        """


        Parameter
        --------

        Returns
        -------
        """
        if isinstance(X, np.ndarray):
            self.feature_names_ = [f"feature_{i}" for i in range(len(X.T))]
        elif isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.values

        X, y = check_X_y(X, y, dtype=object)

        if self.criterion_ is None:
            self.criterion_ = {}
        # NOTE: the key values are numeric indexes of the dataset
        self.unique_levels_ = {d: np.unique(X[:, d]) for d in range(len(X.T))}

        self.tree_ = self._make_tree(X, y)
        return self

    def predict_prob(self, X):
        """


        Parameter
        --------

        Returns
        -------
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        pred_node = self.predict(X)
        return pred_node.shape / pred_node.n_samples

    def predict(self, X):
        """


        Parameter
        --------

        Returns
        -------
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        node = self.tree_
        while not node.is_leaf:
            query_branch = X[node.feature]
            node = node.children[query_branch]
        return node

    def score(self, X, y):
        """


        Parameter
        --------

        Returns
        -------
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y)

        y_hat = [self.predict(X[d, :].feature for d in range(len(X)))]
        return accuracy_score(y, y_hat)

    def _entropy(self, X, y):
        """


        Parameter
        --------

        Returns
        -------
        """
        proba = np.unique(y, return_counts=True)[1] / len(X)
        return -np.sum(proba * np.log2(proba))

    def _cond_entropy(self, X, y, d):
        """


        Parameter
        --------

        Returns
        -------
        """
        weight = np.unique(X[:, d], return_counts=True)[1] / len(X)
        purity = [
            self._entropy(X[X[:, d] == t], y[X[:, d] == t]) for t in np.unique(X[:, d])
        ]
        return np.dot(weight, purity)

    def _compute_information_gain(self, X, y, d):
        """


        Parameter
        --------

        Returns
        -------
        """
        gain = self._entropy(X, y) - self._cond_entropy(X, y, d)
        return gain

    def _partition_data(self, X, y, d, level):
        """


        Parameter
        --------

        Returns
        -------
        """
        idx = X[:, d] == level
        return np.delete(X[idx], d, axis=1), y[idx], level

    def _make_tree(self, X, y, /, *, parent_y=None, branch=None, depth=0):
        """


        Parameter
        --------

        Returns
        -------
        """

        def make_node(value, *, deep=False):
            node = DecisionNode(
                feature=value,
                branch=branch,
                depth=depth,
                shape=np.unique(y, return_counts=True)[1],
            )
            return deepcopy(node) if deep else node

        if len(np.unique(y)) == 1:
            return make_node(mode(y))
        if not X.size:
            return make_node(mode(parent_y))
        if np.all(X == X[0]):
            return make_node(mode(y))
        if self.criterion_.get("max_depth", np.inf) <= depth:
            return make_node(mode(y))
        if self.criterion_.get("min_samples_split", -np.inf) >= len(X):
            return make_node(mode(y))

        # `max_info_gain` represents the feature index that provides the optimal split
        max_info_gain = np.argmax(
            [self._compute_information_gain(X, y, d) for d in range(len(X.T))]
        )

        if self.criterion_.get("min_gain", -np.inf) >= max_info_gain:
            return make_node(mode(y))

        split_feature = self.feature_names_[max_info_gain]
        decision_node = make_node(split_feature, deep=True)

        # if is_numeric_dtype(X[best_feature]):
        #     decision_node.threshold = self.n_thresholds[best_feature]

        levels = [
            self._partition_data(X, y, max_info_gain, level)
            for level in self.unique_levels_[max_info_gain]
        ]

        for *partition_data, level in levels:
            decision_node.add(
                self._make_tree(
                    *partition_data, parent_y=y, branch=level, depth=depth + 1
                )
            )

        return decision_node
