from __future__ import annotations

from copy import deepcopy
from operator import ge, lt
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


class BaseDecisionTree:
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
        return proba.shape / proba.n_samples


class DecisionTreeClassifier(BaseDecisionTree, BaseEstimator, ClassifierMixin):
    def __init__(self, *, criterion=None):
        super().__init__()
        self.criterion_ = criterion

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
        self.unique_levels_ = {
            feature: np.unique(X[:, feature]) for feature in range(len(X.T))
        }
        # self.unique_levels_ = {
        #     feature: ((lt, ge) if is_numeric_dtype(X[:, feature]) else np.unique(X[:, feature]))
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

        proba = np.proba(y)
        print(
            f"H({set(np.unique(y))})",
            f"- ({proba} * log2({proba}))",
            f"{-np.sum(proba * np.log2(proba))}",
            sep="\n\t = ",
        )
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
        print(
            f"IG({self.feature_names_[feature]}) = H({set(np.unique(y))}) - rem({self.feature_names_[feature]})"
        )

        if is_numeric_dtype(X[feature, :]):
            max_gain, self.n_thresholds_[feature] = self._compute_optimal_threshold(
                X, y, feature
            )
        else:
            a = self._entropy(y)
            print(
                f"rem({self.feature_names_[feature]}, {np.unique(X[:, feature])})",
                end="",
            )
            weight = np.proba(X[:, feature])
            impurity = []
            for level in np.unique(X[:, feature]):
                print("\n\t = ", end="")
                impurity.append(self._entropy(y[X[:, feature] == level]))
            # impurity = [
            #     self._entropy(y[X[:, feature] == level])
            #     for level in np.unique(X[:, feature])
            # ]
            print(f"\t = {weight} * {impurity} = {np.dot(weight, impurity)}")
            print("=", self._entropy(y) - np.dot(weight, impurity))
            print("-" * 50)
            max_gain = a - np.dot(weight, impurity)
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
        #         *list(map(lambda f: f[level(X[:, feature], threshold)], [X, y])),
        #         f"{'<' if level is lt else '>='} {threshold}",
        #     )
        mask = X[:, feature] == level
        return np.delete(X[mask], feature, axis=1), y[mask], level

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
        max_gain_feature = np.argmax(
            [
                self._compute_information_gain(X, y, feature)
                for feature in range(len(X.T))
            ]
        )

        if self.criterion_.get("min_gain", -np.inf) >= max_gain_feature:
            return make_node(mode(y))

        split_feature = self.feature_names_[max_gain_feature]
        split_node = make_node(split_feature, deep=True)

        # if is_numeric_dtype(X[:, max_gain_feature]):
        #     split_node.threshold = self.n_thresholds_[max_gain_feature]

        levels = [
            self._partition_data(X, y, max_gain_feature, level, split_node.threshold)
            for level in self.unique_levels_[max_gain_feature]
        ]

        for *partition_data, level in levels:
            split_node.add(
                self._make_tree(
                    *partition_data, parent_y=y, branch=level, depth=depth + 1
                )
            )

        return split_node


class DecisionTreeRegressor(BaseDecisionTree, BaseEstimator, RegressorMixin):
    def __init__(self):
        super().__init__()

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

    def _make_tree(self, X, y, /, *, branch=None, depth=0):
        """


        Parameter
        --------

        Returns
        -------
        """
        raise NotImplementedError


class ParallelDecisionTreeClassifier(DecisionTreeClassifier):
    def _make_tree(
        self, X, y, /, comm=WORLD_COMM, *, parent_y=None, branch=None, depth=0
    ):
        """


        Parameter
        --------

        Returns
        -------
        """
        raise NotImplementedError
