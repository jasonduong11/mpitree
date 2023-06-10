from __future__ import annotations

DEBUG = True

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from statistics import mode

import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._node import DecisionNode

WORLD_COMM = MPI.COMM_WORLD
WORLD_RANK = WORLD_COMM.Get_rank()
WORLD_SIZE = WORLD_COMM.Get_size()


def is_numeric_dtype(arr):
    try:
        arr.astype(np.float64)
        return True
    except ValueError:
        return False


def proba(X):
    """


    Parameter
    --------

    Returns
    -------
    """
    return np.unique(X, return_counts=True)[1] / len(X)


def split_mask(X, mask):
    return [[X[mask], X[~mask]]]


def Get_cyclic_dist(comm: MPI.Intracomm = None, *, n_block: int = 1) -> MPI.Intracomm:
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


np.proba = proba
np.split_mask = split_mask
MPI.Get_cyclic_dist = Get_cyclic_dist


class BaseDecisionTree(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *, max_depth, min_samples_split, min_gain, estimator_type):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        DecisionNode._estimator_type = estimator_type

    def __iter__(self):
        """Perform a depth-first search on the decision tree estimator.

        The traversal starts at the root node and recursively traverses
        across all its children in a sorted order. In each level in the
        decision tree estimator, non-leaf always precede leaf nodes.

        Parameters
        ----------
        None

        Yields
        ------
        DecisionNode

        Notes
        -----
        Since the tree data structure is a DAG (Directed Acyclic Graph),
        we do not maintain a list of visited nodes for nodes already and/or
        yet-to-be explored.
        """
        check_is_fitted(self)

        frontier = [self.tree_]
        while frontier:
            node = frontier.pop()
            yield node
            frontier.extend(sorted(node.children.values()))

    def __repr__(self):
        """


        Parameter
        --------

        Returns
        -------
        """
        check_is_fitted(self)
        return "\n".join(map(str, self))

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

    def predict_prob(self, X):
        """


        Parameter
        --------

        Returns
        -------
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        proba = self.predict(X)
        return proba.value / proba.n_samples


class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    def __init__(self, *, max_depth=np.inf, min_samples_split=1, min_gain=-1):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_gain=min_gain,
            estimator_type="classifier",
        )

    def fit(self, X, y):
        """


        Parameter
        --------

        Returns
        -------
        """
        if isinstance(X, np.ndarray):
            # default feature names if no feature names were given
            self.feature_names_ = [f"feature_{i}" for i in range(len(X.T))]
        elif isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.values

        X, y = check_X_y(X, y, dtype=object)

        # NOTE: the key values are numeric indexes of the dataset
        self.unique_levels_ = {
            feature: np.unique(X[:, feature]) for feature in range(len(X.T))
        }

        # self.unique_levels_ = {
        #     feature: (("True", "False") if is_numeric_dtype(X[:, feature]) else np.unique(X[:, feature]))
        #     for feature in range(len(X.T))
        # }
        # self.n_thresholds_ = {}

        self.tree_ = self._make_tree(X, y)
        return self

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

    def _entropy(self, y):
        """


        Parameter
        --------

        Returns
        -------
        """

        if DEBUG:
            proba = np.proba(y)
            print(
                f"\t = H({set(np.unique(y))})",
                f"- {proba.round(3)} * log({proba.round(3)})",
                f"[{np.sum(proba * np.log2(proba)).round(3)}]",
                sep="\n\t\t = ",
            )
            return -np.sum(proba * np.log2(proba))
        else:
            proba = np.proba(y)
            return -np.sum(proba * np.log2(proba))

    # def _compute_optimal_threshold(self, X, y, feature):
    #     arr = np.column_stack((X, y))
    #     arr = arr[arr[:, feature].argsort()]

    #     thresholds = []
    #     for i, j in [(i, i+1) for i, (p, q) in enumerate(zip(y, y[1:])) if p != q]:
    #         thresholds.append(
    #             np.mean(X[i:j+1, feature].astype(np.float64))
    #         )

    #     print(f"Possible Thresholds: {thresholds}")

    #     A = []
    #     for threshold in thresholds:
    #         levels = np.split_mask(X, X[:, feature] < threshold)
    #         # print(f"Levels: {levels}")

    #         weights = [len(l) / len(arr) for l in levels]
    #         print(f"Weights: {weights}")

    #         total_entropy = self._entropy(arr[-1])
    #         print(f"Total Entropy: {total_entropy}")

    #         cond_entropy = [self._entropy(level[-1]) for level in levels]
    #         print(f"Cond Entropy: {cond_entropy}")

    #         rem = np.dot(weights, cond_entropy)
    #         A.append(total_entropy - rem)

    #     print(f"A: {A}")

    #     t_hat = thresholds[np.argmax(A)]
    #     print(f"Optimal Threshold: {t_hat}")

    #     return max(A), t_hat

    def _compute_information_gain(self, X, y, feature):
        """


        Parameter
        --------

        Returns
        -------
        """
        if is_numeric_dtype(X[feature, :]):
            max_gain, self.n_thresholds_[feature] = self._compute_optimal_threshold(
                X, y, feature
            )
        else:
            if DEBUG:
                print(
                    f"\nIG({self.feature_names_[feature]}) = H({set(np.unique(y))}) - rem({self.feature_names_[feature]})"
                )
                total_entropy = self._entropy(y)

                print(f"\t = rem({self.feature_names_[feature]})")

                weight = np.proba(X[:, feature])
                impurity = []
                for level in np.unique(X[:, feature]):
                    a = self._entropy(y[X[:, feature] == level])
                    impurity.append(a)

                print(
                    f"\t = {weight.round(3)} x {np.array(impurity).round(3)}",
                    f"[{np.dot(weight, impurity).round(3)}]",
                    sep="\n\t\t = ",
                )

                max_gain = total_entropy - np.dot(weight, impurity)
                print(f"\t = [{max_gain.round(3)}]")
            else:
                weight = np.proba(X[:, feature])
                impurity = [
                    self._entropy(y[X[:, feature] == level])
                    for level in np.unique(X[:, feature])
                ]
                max_gain = self._entropy(y) - np.dot(weight, impurity)
        return max_gain

    def _partition_data(self, X, y, feature, level, threshold):
        """


        Parameter
        --------

        Returns
        -------
        """
        # if threshold is not None:
        #     return (
        #         *list(
        #             map(
        #                 lambda f: f[(X[:, feature] < threshold)]
        #                 if level
        #                 else f[(X[:, feature] >= threshold)],
        #                 [X, y],
        #             )
        #         ),
        #         level,
        #     )

        if DEBUG:
            mask = X[:, feature] == level
            a = np.delete(X[mask], feature, axis=1)
            print(50 * "-")
            print(np.column_stack((a, y[mask])))
            print("+")
            print(X[mask])
            return a, y[mask], level
        else:
            mask = X[:, feature] == level
            return np.delete(X[mask], feature, axis=1), y[mask], level

    def _make_tree(self, X, y, /, *, parent=None, branch=None):
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
        # NOTE: parent param is None during start
        # NOTE: parent depth + 1 is the current depth
        if parent and self.max_depth <= parent.depth + 1:
            return make_node(mode(y))
        if self.min_samples_split >= len(X):
            return make_node(mode(y))

        info_gains = [self._compute_information_gain(X, y, feature) for feature in range(len(X.T))]
        split_feature_idx = np.argmax(info_gains)

        # NOTE: `min_gain` hyperparameter is default to -1 bc the domain is [0, inf]
        if self.min_gain >= info_gains[split_feature_idx]:
            return make_node(mode(y))

        split_feature = self.feature_names_[split_feature_idx]
        split_node = make_node(split_feature, deep=True)

        # if is_numeric_dtype(X[:, max_gain_feature]):
        #     split_node.threshold = self.n_thresholds_[max_gain_feature]

        if DEBUG:
            print(f"\nSplit on Feature: {split_feature}")
            print(50 * "-")
            print(X)

        levels = [
            self._partition_data(X, y, split_feature_idx, level, split_node.threshold)
            for level in self.unique_levels_[split_feature_idx]
        ]

        for *partition_data, level in levels:
            split_node.add(
                self._make_tree(*partition_data, parent=split_node, branch=level)
            )

        return split_node


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    def __init__(self, *, max_depth=np.inf, min_samples_split=1, min_gain=-1):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_gain=min_gain,
            estimator_type="regressor",
        )

    def fit(self, X, y):
        """


        Parameter
        --------

        Returns
        -------
        """
        X, y = check_X_y(X, y, y_numeric=True)
        return self

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
        return mean_squared_error(y, y_hat, squared=False)

    def _make_tree(self, X, y, /, *, parent=None, branch=None):
        """


        Parameter
        --------

        Returns
        -------
        """
        raise NotImplementedError


class ParallelDecisionTreeClassifier(DecisionTreeClassifier):
    def _make_tree(self, X, y, /, comm=WORLD_COMM, *, parent=None, branch=None):
        """


        Parameter
        --------

        Returns
        -------
        """
        raise NotImplementedError
