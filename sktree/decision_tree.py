from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from statistics import mode

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._node import DecisionNode
from ._util import is_numeric_dtype

DEBUG = 1


class BaseDecisionTree(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *, max_depth, min_samples_split, min_gain, estimator_type):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        DecisionNode._estimator_type = estimator_type

    def __iter__(self):
        """Perform a depth-first search on a decision tree estimator.

        The iterative traversal starts at the root decision node and
        explores as far as possible from its sorted children.

        Yields
        ------
        DecisionNode

        Notes
        -----
        Since the decision tree estimator is a DAG (Directed Acyclic Graph)
        data structure, we do not maintain a list of visited nodes for
        nodes already and yet to be explored. The ordering of children
        decision nodes in the stack are as follows, non-leaf always
        precedes leaf nodes.

        See Also
        --------
        DecisionNode.__lt__ : An ordering for decision nodes.
        """
        check_is_fitted(self)

        frontier = [self.tree_]
        while frontier:
            node = frontier.pop()
            yield node
            frontier.extend(sorted(node.children.values()))

    def __str__(self):
        """Short Summary

        Extended Summary

        Returns
        -------
        str
        """
        check_is_fitted(self)
        return "\n".join(map(str, self))

    def _tree_walk(self, x):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        x : np.ndarray, ndim=1

        Returns
        -------
        DecisionNode
        """
        check_is_fitted(self)

        node = self.tree_
        while not node.is_leaf:
            # np.apply_along_axis removes feature names, so we cannot
            # index using the current tree node feature name, instead,
            # we can retreive the index of the current tree node feature
            # name to get the query feature value
            feature_idx = self.feature_names.index(node.feature)
            query_branch = x[feature_idx]
            if isinstance(query_branch, float | int):
                node = node.left if query_branch < node.threshold else node.right
            else:
                node = node.children[query_branch]
        return node

    def predict(self, X):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        x : np.ndarray, ndim=2

        Returns
        -------
        np.ndarray
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        return np.apply_along_axis(lambda x: self._tree_walk(x).feature, axis=1, arr=X)

    def predict_proba(self, X):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        x : np.ndarray, ndim=2

        Returns
        -------
        np.ndarray
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        proba = np.apply_along_axis(self._tree_walk, axis=1, arr=X)

        output = []
        for node in proba:
            output.append(node.value / node.n_samples)
        return output

        # NOTE: not working because of inconsisent dimensions
        # return np.vectorize(lambda node: node.value / node.n_samples)(proba)


class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    def __init__(self, *, max_depth=np.inf, min_samples_split=2, min_gain=-1):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_gain=min_gain,
            estimator_type="classifier",
        )

    def fit(self, X, y):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray, ndim=2
        y : np.ndarray, ndim=1

        Returns
        -------
        DecisionTreeClassifier
        """
        # NOTE: `feature_names` attribute is a `list` type bc from `index` method
        if isinstance(X, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.values.tolist()

        X, y = check_X_y(X, y, dtype=object)

        self.optimal_threshold_ = None

        # NOTE: the key values are numeric indexes of the dataset
        self.unique_levels_ = {
            feature_idx: (
                ("True", "False")
                if is_numeric_dtype(X[:, feature_idx])
                else np.unique(X[:, feature_idx])
            )
            for feature_idx in range(X.shape[1])
        }

        self.tree_ = self._tree_builder(X, y)
        return self

    def _entropy(self, x):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        x : np.ndarray, ndim=1

        Returns
        -------
        float
        """
        if DEBUG:
            proba = np.proba(x)
            print(
                f"\t = H({set(np.unique(x))})",
                f"- {proba.round(3)} * log({proba.round(3)})",
                f"[{np.sum(proba * np.log2(proba)).round(3)}]",
                sep="\n\t\t = ",
            )
            return -np.sum(proba * np.log2(proba))
        else:
            proba = np.proba(x)
            return -np.sum(proba * np.log2(proba))

    def _compute_optimal_threshold(self, X, y, feature_idx):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray, ndim=2
        y : np.ndarray, ndim=1
        feature_idx : int

        Returns
        -------
        max_gain : The maximum information gain value.
        t_hat : The optimal threshold value.
        """

        def cost(x):
            threshold = x[-1]
            mask = X[:, feature_idx] < threshold
            levels = np.split_mask(X, mask)
            weights = [len(level) / len(X) for level in levels]
            impurity = [self._entropy(y[mask]), self._entropy(y[~mask])]
            return np.dot(weights, impurity)

        costs = np.apply_along_axis(cost, axis=1, arr=X)
        idx = np.argmin(costs)
        t_hat = X[idx, -1]

        min_cost = min(costs)
        max_gain = self._entropy(y) - min_cost

        return max_gain, t_hat

    def _compute_information_gain(self, X, y, feature_idx):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray, ndim=2
        y : np.ndarray, ndim=1
        feature_idx : int

        Returns
        -------
        max_gain : The maximm information gain value.
        """
        if is_numeric_dtype(X[:, feature_idx]):
            max_gain, self.optimal_threshold_ = self._compute_optimal_threshold(
                X, y, feature_idx
            )
        else:
            if DEBUG:
                print(
                    f"\nIG({self.feature_names[feature_idx]}) =",
                    f"H({set(np.unique(y))}) - rem({self.feature_names[feature_idx]})",
                )
                total_entropy = self._entropy(y)

                print(f"\t = rem({self.feature_names[feature_idx]})")

                weight = np.proba(X[:, feature_idx])
                impurity = []
                for level in np.unique(X[:, feature_idx]):
                    a = self._entropy(y[X[:, feature_idx] == level])
                    impurity.append(a)

                print(
                    f"\t = {weight.round(3)} x {np.array(impurity).round(3)}",
                    f"[{np.dot(weight, impurity).round(3)}]",
                    sep="\n\t\t = ",
                )

                max_gain = total_entropy - np.dot(weight, impurity)
                print(f"\t = [{max_gain.round(3)}]")
            else:
                weight = np.proba(X[:, feature_idx])
                impurity = [
                    self._entropy(y[X[:, feature_idx] == level])
                    for level in np.unique(X[:, feature_idx])
                ]
                max_gain = self._entropy(y) - np.dot(weight, impurity)
        return max_gain

    def _partition_data(self, X, y, split_feature_idx, level, threshold):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray, ndim=2
        y : np.ndarray, ndim=1
        split_feature_idx : int
        level : str
        threshold : float or None

        Returns
        -------
        """
        if threshold is not None:
            mask = X[:, split_feature_idx] < threshold
            return [
                (tree, y[mask], l)
                for tree, l in zip(np.split_mask(X, mask), ["True", "False"])
            ]

        if DEBUG:
            mask = X[:, split_feature_idx] == level
            a = np.delete(X[mask], split_feature_idx, axis=1)
            print(50 * "-")
            print(np.column_stack((a, y[mask])))
            print("+")
            print(X[mask])
            return a, y[mask], level
        else:
            mask = X[:, split_feature_idx] == level
            return np.delete(X[mask], split_feature_idx, axis=1), y[mask], level

    def _tree_builder(self, X, y, /, *, parent=None, branch=None):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray, ndim=2
        y : np.ndarray, ndim=1
        parent : DecisionTreeClassifier, default=None
        branch : str, default=None

        Returns
        -------
        DecisionNode
        """

        def make_node(value, *, deep=False):
            node = DecisionNode(
                feature=value,
                branch=branch,
                parent=parent,
                state=np.column_stack((X, y)),
            )
            return deepcopy(node) if deep else node

        if len(np.unique(y)) == 1:
            return make_node(mode(y))
        if not X.size:
            return make_node(mode(parent.y))
        if np.all(X == X[0]):
            return make_node(mode(y))

        # NOTE: also check if max_depth is 0
        # NOTE: parent param is None during start
        # NOTE: parent depth + 1 is the current depth
        # if not self.max_depth or parent and self.max_depth <= parent.depth + 1:
        #     return make_node(mode(y))
        if self.min_samples_split > len(X):
            return make_node(mode(y))

        info_gains = [
            self._compute_information_gain(X, y, feature_idx)
            for feature_idx in range(X.shape[1])
        ]
        split_feature_idx = np.argmax(info_gains)

        # NOTE: `min_gain` hyperparameter is default to -1 bc the domain is [0, inf]
        # if self.min_gain >= info_gains[split_feature_idx]:
        #     return make_node(mode(y))

        split_feature = self.feature_names[split_feature_idx]
        split_node = make_node(split_feature, deep=True)

        if is_numeric_dtype(X[:, split_feature_idx]):
            split_node.threshold = self.optimal_threshold_

        if DEBUG:
            print(f"\nSplit on Feature: {split_feature}")
            print(50 * "-")
            print(X)

        levels = [
            self._partition_data(X, y, split_feature_idx, level, split_node.threshold)
            for level in self.unique_levels_[split_feature_idx]
        ]

        if is_numeric_dtype(X[:, split_feature_idx]):
            levels = levels[0]

        for *partition_data, level in levels:
            split_node.add(
                self._tree_builder(*partition_data, parent=split_node, branch=level)
            )

        return split_node

    def score(self, X, y):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray, ndim=2
        y : np.ndarray, ndim=1

        Returns
        -------
        float
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y)

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    def __init__(self, *, max_depth=np.inf, min_samples_split=1, min_gain=-1):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_gain=min_gain,
            estimator_type="regressor",
        )

    def fit(self, X, y):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray, ndim=2
        y : np.ndarray, ndim=1

        Returns
        -------
        DecisionTreeRegressor
        """
        X, y = check_X_y(X, y, y_numeric=True)
        return self

    def _tree_builder(self, X, y, /, *, parent=None, branch=None):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray, ndim=2
        y : np.ndarray, ndim=1
        parent : DecisionTreeRegressor, default=None
        branch : str, default=None

        Returns
        -------
        DecisionNode
        """
        raise NotImplementedError

    def score(self, X, y):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray, ndim=2
        y : np.ndarray, ndim=1

        Returns
        -------
        float
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y, y_numeric=True)

        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred, squared=False)


class ParallelDecisionTreeClassifier(DecisionTreeClassifier):
    from mpi4py import MPI

    WORLD_COMM = MPI.COMM_WORLD
    WORLD_RANK = WORLD_COMM.Get_rank()
    WORLD_SIZE = WORLD_COMM.Get_size()

    def Get_cyclic_dist(
        self, comm: MPI.Intracomm = None, *, n_block: int = 1
    ) -> MPI.Intracomm:
        """Schedules processes in a round-robin fashion.

        Parameters
        ----------
        comm : MPI.Intracomm, default=None
        n_block : int, default=1

        Returns
        -------
        MPI.Intracomm
        """
        rank = comm.Get_rank()
        key, color = divmod(rank, n_block)
        return comm.Split(color, key)

    def _tree_builder(self, X, y, /, comm=WORLD_COMM, *, parent=None, branch=None):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray, ndim=2
        y : np.ndarray, ndim=1
        comm : MPI.Intracomm, default=WORLD_COMM
        parent : DecisionTreeRegressor, default=None
        branch : str, default=None

        Returns
        -------
        DecisionNode
        """
        raise NotImplementedError
