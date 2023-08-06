""""""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from statistics import mode

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._node import DecisionNode


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


np.proba = proba
np.split_mask = split_mask


class BaseDecisionTree(BaseEstimator, metaclass=ABCMeta):
    """Short Summary

    Extended Summary

    Parameters
    ----------
    max_depth : int, default=None
        A hyperparameter that upper bounds the number of splits.

    min_samples_split : int, default=2
        A hyperparameter that lower bounds the number of samples required to
        split.
    """

    @abstractmethod
    def __init__(self, *, max_depth, min_samples_split):
        self.max_depth = max_depth  # [0, inf)
        self.min_samples_split = min_samples_split  # [2, inf)

    def __str__(self):
        """Export a text-based visualization of a decision tree estimator.

        The function is a wrapper function for the overloaded `iter`
        method that displays each string-formatted `DecisionNode` in a
        decision tree estimator.

        Returns
        -------
        str
        """
        check_is_fitted(self)
        return "\n".join(map(str, self))

    def __iter__(self):
        """Perform a depth-first search on a decision tree estimator.

        The iterative traversal starts at the root decision node, explores
        as deep as possible from its prioritized children, and backtracks
        after reaching a leaf decision node.

        Yields
        ------
        DecisionNode

        Notes
        -----
        Since a decision tree estimator is a DAG (Directed Acyclic Graph)
        data structure, we do not maintain a list of visited nodes for
        nodes already explored or the frontier for nodes already pushed,
        but yet to be explored (i.e., duplicate nodes).

        The ordering of decision nodes at each level in the stack is as
        follows, non-leaf always precedes leaf nodes. This provides the
        minimum number of disjoint branch components at each level of a
        decision tree estimator.
        """
        check_is_fitted(self)

        frontier = [self.tree_]
        while frontier:
            tree_node = frontier.pop()
            yield tree_node
            frontier.extend(
                sorted(tree_node.children.values(), key=lambda n: not n.is_leaf)
            )

    def _decision_paths(self, X):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D test feature matrix with shape (n_samples, n_features) of
            either or both categorical and numerical values.

        Returns
        -------
        np.ndarray
            An array of predicated leaf decision nodes.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        def tree_walk(x) -> DecisionNode:
            """Traverse a decision tree estimator from root to some leaf node.

            Decision nodes are queried based on the respective feature value
            from `x` at each level until some leaf decision node is reached.

            Parameters
            ----------
            x : np.ndarray
                1D test instance array of shape (1, n_features) from a feature
                matrix `X`.

            Returns
            -------
            DecisionNode

            Notes
            -----
            The `np.apply_along_axis` function does not consider feature names,
            so we cannot lookup using the current node feature name. Instead,
            we can retrieve the index of the current node feature name to
            lookup the query feature value.
            """
            tree_node = self.tree_
            while not tree_node.is_leaf:
                feature_idx = self.feature_names_.index(tree_node.feature)
                query_level = x[feature_idx]

                if query_level <= tree_node.threshold:
                    tree_node = tree_node.left
                else:
                    tree_node = tree_node.right

            return tree_node

        return np.apply_along_axis(tree_walk, axis=1, arr=X)

    @abstractmethod
    def _make_tree(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        parent: DecisionNode,
        branch: str | float,
        depth: int,
    ) -> DecisionNode:
        ...

    def predict(self, X):
        """Return the predicated leaf decision node.

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D test feature matrix with shape (n_samples, n_features) of
            either or both categorical and numerical values.

        Returns
        -------
        np.ndarray
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        return np.vectorize(lambda n: n.feature)(self._decision_paths(X))


class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    def __init__(self, *, max_depth=None, min_samples_split=2):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)

    def fit(self, X, y):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of either
            or both categorical and numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of categorical values.

        Returns
        -------
        DecisionTreeClassifier

        Notes
        -----
        The `feature_names` attribute is a `list` type so we can use the
        `index` method.
        """
        if isinstance(X, list):
            self.feature_names_ = [f"feature_{i}" for i in range(len(X[0]))]
        elif isinstance(X, np.ndarray):
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        elif isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.values.tolist()
        else:
            raise Exception("could not find type for `feature_names_` var")

        X, y = check_X_y(X, y, dtype=object)

        self.classes_ = np.unique(y)
        self.n_thresholds_ = {}

        self.tree_ = self._make_tree(X, y)
        return self

    def _entropy(self, x):
        """Measure the impurity on some feature.

        The parameter `x` is assumed to represent values for some
        particular feature (including the label or target feature) on some
        dataset `X`.

        Parameters
        ----------
        x : np.ndarray
            1D feature column array of shape (n_samples,).

        Returns
        -------
        float
        """
        proba = np.proba(x)
        return -np.sum(proba * np.log2(proba))

    def _compute_optimal_threshold(self, X, y, feature_idx):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of either
            or both categorical and numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of categorical values.

        feature_idx : int

        Returns
        -------
        max_gain : The maximum information gain value.
        t_hat : The optimal threshold value.
        """

        order = X[:, feature_idx].argsort()

        X_sorted = X[order]
        y_sorted = y[order]

        thresholds = []
        for i in range(len(y_sorted) - 1):
            if y_sorted[i] != y_sorted[i + 1]:
                thresholds.append(np.mean((X_sorted[i : i + 2, feature_idx])))

        def cost(t):
            mask = X[:, feature_idx] <= t
            levels = np.split_mask(X, mask)
            weights = np.array([len(level) / len(X) for level in levels])
            impurity = np.array([self._entropy(y[mask]), self._entropy(y[~mask])])
            return np.dot(weights, impurity)

        costs = [cost(t) for t in thresholds]
        t_hat = thresholds[np.argmin(costs)]

        min_cost = min(costs)
        max_gain = self._entropy(y) - min_cost

        assert (
            X[:, feature_idx].min()
            <= thresholds[np.argmin(costs)]
            <= X[:, feature_idx].max()
        )

        return max_gain, t_hat

    def _compute_information_gain(self, X, y, feature_idx):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of either
            or both categorical and numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of categorical values.

        feature_idx : int

        Returns
        -------
        max_gain : The maximum information gain value.
        """
        max_gain, self.n_thresholds_[feature_idx] = self._compute_optimal_threshold(
            X, y, feature_idx
        )
        return max_gain

    def _make_tree(self, X, y, *, parent=None, branch=None, depth=0):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of either
            or both categorical and numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of categorical values.

        parent : DecisionTreeClassifier, default=None

        branch : str, default=None

        depth : int, default=0

        Returns
        -------
        DecisionNode
        """

        def make_node(value):
            return deepcopy(
                DecisionNode(
                    feature=value,
                    branch=branch,
                    parent=parent,
                    target=y,
                    classes=self.classes_,
                )
            )

        if len(np.unique(y)) == 1:
            return make_node(mode(y))
        if not X.size:
            return make_node(mode(parent.target))
        if np.all(X == X[0]):
            return make_node(mode(y))
        if self.max_depth is not None and self.max_depth == depth:
            return make_node(mode(y))
        if self.min_samples_split > len(X):
            return make_node(mode(y))

        split_feature_idx = np.argmax(
            [
                self._compute_information_gain(X, y, feature_idx)
                for feature_idx in range(X.shape[1])
            ]
        )

        split_node = make_node(self.feature_names_[split_feature_idx])
        split_node.threshold = self.n_thresholds_[split_feature_idx]

        mask = X[:, split_feature_idx] <= split_node.threshold

        if not all(i.size for i in np.split_mask(X, mask)):
            print(split_node.threshold)
            print(X)
            raise AssertionError()

        if not all(i.size for i in np.split_mask(y, mask)):
            print(split_node.threshold)
            print(y)
            raise AssertionError()

        n_subtrees = zip(np.split_mask(X, mask), np.split_mask(y, mask), ("<=", ">"))

        for X, y, branch in n_subtrees:
            subtree = self._make_tree(
                X=X,
                y=y,
                parent=split_node,
                branch=branch,
                depth=depth + 1,
            )
            split_node.children[branch] = subtree

        return split_node

    def predict_proba(self, X):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D test feature matrix with shape (n_samples, n_features) of either
            or both categorical and numerical values.

        Returns
        -------
        np.ndarray
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        def compute_class_proba(tree_node):
            if tree_node.n_samples == 0:
                return tree_node.parent.value / tree_node.parent.n_samples
            return tree_node.value / tree_node.n_samples

        return np.vectorize(compute_class_proba, signature="()->(n)")(
            self._decision_paths(X)
        )

    def score(self, X, y):
        """Evaluate the performance of a decision tree classifier.

        The metric used to evaluate the performance of a decision tree
        classifier is `sklearn.metrics.accuracy_score`.

        Parameters
        ----------
        X : np.ndarray
            2D test feature matrix with shape (n_samples, n_features) of
            either or both categorical and numerical values.

        y : np.ndarray
            1D test target array with shape (n_samples,) of categorical values.

        Returns
        -------
        float
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y, dtype=object)

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class ParallelDecisionTreeClassifier(DecisionTreeClassifier):
    pass


#     """Short Summary

#     Extended Summary
#     """

#     from mpi4py import MPI

#     WORLD_COMM = MPI.COMM_WORLD
#     WORLD_RANK = WORLD_COMM.Get_rank()
#     WORLD_SIZE = WORLD_COMM.Get_size()

#     def Get_cyclic_dist(
#         self, comm: MPI.Intracomm = None, *, n_block: int = 1
#     ) -> MPI.Intracomm:
#         """Schedules processes in a round-robin fashion.

#         Parameters
#         ----------
#         comm : MPI.Intracomm, default=None
#         n_block : int, default=1

#         Returns
#         -------
#         MPI.Intracomm
#         """
#         rank = comm.Get_rank()
#         key, color = divmod(rank, n_block)
#         return comm.Split(color, key)

#     def _make_tree(
#         self, *, X, y, parent=None, branch=None, depth=0, comm=WORLD_COMM
#     ):
#         """Short Summary

#         Extended Summary

#         Parameters
#         ----------
#         X : np.ndarray
#             2D feature matrix with shape (n_samples, n_features) of either
#             or both categorical and numerical values.

#         y : np.ndarray
#             1D target array with shape (n_samples,) of numerical values.

#         comm : MPI.Intracomm, default=WORLD_COMM

#         parent : DecisionTreeRegressor, default=None

#         branch : str, default=None

#         depth : int, default=0

#         Returns
#         -------
#         DecisionNode
#         """
#         raise NotImplementedError
