from __future__ import annotations

from copy import deepcopy
from statistics import mode

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._node import DecisionNode


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *, criterion=None):
        super().__init__()
        self.criterion_ = criterion

    def __iter__(self, other: DecisionNode = None):
        check_is_fitted(self)

        if not other:
            other = self.root_

        yield other
        for child_node in other.children.values():
            yield from self.__iter__(child_node)

    def __repr__(self):
        check_is_fitted(self)
        return "\n".join(map(str, self))

    def fit(self, X, y):
        self.feature_names_ = (
            range(len(X.T)) if isinstance(X, np.ndarray) else X.columns.values
        )
        X, y = check_X_y(X, y, dtype=object)

        if self.criterion_ is None:
            self.criterion_ = {}
        # NOTE: the key values are numeric indexes of the dataset
        self.n_levels_ = {d: np.unique(X[:, d]) for d in range(len(X.T))}

        self.root_ = self._make_tree(X, y)
        return self

    def predict_prob(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        pred_node = self.predict(X)
        return pred_node.shape / pred_node.samples

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        node = self.root_
        while not node.is_leaf:
            query_branch = X[node.feature]
            node = node.children[query_branch]
        return node

    def score(self, X, y):
        check_is_fitted(self)
        X, y = check_X_y(X, y)

        y_hat = [self.predict(X[d, :].feature for d in range(len(X)))]
        return accuracy_score(y, y_hat)

    def entropy(self, X, y):
        proba = np.unique(y, return_counts=True)[1] / len(X)
        return -np.sum(proba * np.log2(proba))

    def cond_entropy(self, X, y, d):
        weight = np.unique(X[:, d], return_counts=True)[1] / len(X)
        purity = [
            self.entropy(X[X[:, d] == t], y[X[:, d] == t]) for t in np.unique(X[:, d])
        ]
        return np.dot(weight, purity)

    def information_gain(self, X, y, d):
        # NOTE: for categorical variables only only
        gain = self.entropy(X, y) - self.cond_entropy(X, y, d)
        return gain

    def partition_data(self, X, y, d, level):
        """NOTE: `np.delete` returns a copy of the array"""
        idx = X[:, d] == level
        return np.delete(X[idx], d, axis=1), y[idx], level

    def _make_tree(self, X, y, /, *, parent_y=None, branch=None, depth=0):
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

        max_gain = np.argmax([self.information_gain(X, y, d) for d in range(len(X.T))])

        if self.criterion_.get("min_gain", -np.inf) >= max_gain:
            return make_node(mode(y))

        best_feature = self.feature_names_[max_gain]
        best_node = make_node(best_feature, deep=True)

        # if is_numeric_dtype(X[best_feature]):
        #     best_node.threshold = self.n_thresholds[best_feature]

        levels = [
            self.partition_data(X, y, max_gain, level)
            for level in self.n_levels_[max_gain]
        ]

        for *partition_data, level in levels:
            best_node.add(
                self._make_tree(
                    *partition_data, parent_y=y, branch=level, depth=depth + 1
                )
            )

        return best_node
