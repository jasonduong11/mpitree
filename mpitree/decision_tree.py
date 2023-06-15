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
    """Short Summary

    Extended Summary
    """

    @abstractmethod
    def __init__(self, *, max_depth, min_samples_split, min_gain, estimator_type):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        DecisionNode._estimator_type = estimator_type

    def __str__(self):
        """Export a text-based visualization of a decision tree estimator.

        The function displays each string-formatted decision node of a
        decision tree estimator in a depth-first search manner.

        Returns
        -------
        str

        See Also
        --------
        DecisionNode.__str__ : Export a string-formatted decision node.
        """
        check_is_fitted(self)
        return "\n".join(map(str, self))

    def __iter__(self):
        """Perform a depth-first search on a decision tree estimator.

        The iterative traversal starts at the root decision node, explores
        as deep as possible from its sorted children, and backtracks after
        reaching a leaf decision node.

        Yields
        ------
        DecisionNode

        Notes
        -----
        Since a decision tree estimator is a DAG (Directed Acyclic Graph)
        data structure, we do not maintain a list of visited nodes for
        nodes already explored or inspect the stack for nodes yet to be
        explored.

        The ordering of children decision nodes at each level in the stack
        is as follows, non-leaf always precedes leaf nodes.

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

    def _tree_walk(self, x) -> DecisionNode:
        """Traverse a decision tree estimator from root to some leaf node.

        Decision nodes are queried based on the respective feature value
        from `x` at each level until some leaf decision node is reached.

        Parameters
        ----------
        x : np.ndarray
            1D instance array of shape (1, n_features) from a feature
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
        check_is_fitted(self)

        def isfloat(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        node = self.tree_
        while not node.is_leaf:
            feature_idx = self.feature_names_.index(node.feature)
            query_branch = x[feature_idx]
            if isfloat(query_branch):
                node = node.left if query_branch <= node.threshold else node.right
            else:
                node = node.children[query_branch]
        return node

    @abstractmethod
    def _tree_builder(self):
        ...

    def predict(self, X):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

        Returns
        -------
        np.ndarray
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        return np.apply_along_axis(lambda x: self._tree_walk(x).feature, axis=1, arr=X)

    def export_graphviz(self):
        from collections import deque

        import graphviz

        def bfs(s):
            queue = deque([s])
            while queue:
                node = queue.popleft()
                children = sorted(node.children.values())
                yield node, children
                queue.extend(children)

        def info(n):
            return "\n".join(
                [
                    f"feature={n.feature}",
                    f"threshold={n.threshold}",
                    f"branch={n.branch}",
                    f"depth={n.depth}",
                    f"value={n.value}={n.n_samples}",
                    f"proba={n.proba.round(2)}",
                    f"is_leaf={n.is_leaf}",
                    str(n.state),
                ]
            )

        net = graphviz.Digraph(graph_attr={"size": "(6,6)"}, node_attr={"shape": "box"})

        for node, children in bfs(self.tree_):
            net.node(info(node))
            for i, child in enumerate(children):
                net.node(info(child))
                net.edge(info(node), info(child), label=list(node.children)[i])

        return net


class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    def __init__(self, *, max_depth=np.inf, min_samples_split=1, min_gain=-1):
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
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

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
        if isinstance(X, np.ndarray):
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        elif isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.values.tolist()

        X, y = check_X_y(X, y, dtype=object)

        self.classes_ = np.unique(y)
        self.possible_thresholds_ = {}

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
        x : np.ndarray
            1D array of shape (n_samples,).

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
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of categorical values.

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
            if DEBUG:
                print(
                    f"{weights} * {impurity}", np.dot(weights, impurity), sep="\n\t = "
                )
            return np.dot(weights, impurity)

        if DEBUG:
            print(f"Cost({self.feature_names_[feature_idx]}) = ")
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
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of categorical values.

        feature_idx : int

        Returns
        -------
        max_gain : The maximm information gain value.
        """
        if is_numeric_dtype(X[:, feature_idx]):
            (
                max_gain,
                self.possible_thresholds_[feature_idx],
            ) = self._compute_optimal_threshold(X, y, feature_idx)
        else:
            if DEBUG:
                print(
                    f"\nIG({self.feature_names_[feature_idx]}) =",
                    f"H({set(np.unique(y))}) - rem({self.feature_names_[feature_idx]})",
                )
                total_entropy = self._entropy(y)

                print(f"\t = rem({self.feature_names_[feature_idx]})")

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

    def _partition_data(self, X, y, feature_idx, level, split_node):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of categorical values.

        feature_idx : int

        level : str

        threshold : float or None

        Returns
        -------
        """
        if split_node.threshold is not None:
            mask = X[:, feature_idx] < split_node.threshold
            return [
                (tree, y[mask if level == "True" else ~mask], level)
                for tree, level in zip(np.split_mask(X, mask), ["True", "False"])
            ]

        if DEBUG:
            mask = X[:, feature_idx] == level
            a = np.delete(X[mask], feature_idx, axis=1)
            print(50 * "-")
            print(np.column_stack((a, y[mask])))
            print("+")
            print(X[mask])
            return a, y[mask], level
        else:
            mask = X[:, feature_idx] == level
            return np.delete(X[mask], feature_idx, axis=1), y[mask], level

    def _tree_builder(self, X, y, /, *, parent=None, branch=None, depth=0):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of categorical values.

        parent : DecisionTreeClassifier, default=None

        branch : str, default=None

        depth : int, default=0

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
                classes=self.classes_,
            )
            return deepcopy(node) if deep else node

        if len(np.unique(y)) == 1:
            return make_node(mode(y))
        if not X.size:
            return make_node(mode(parent.y))
        if np.all(X == X[0]):
            return make_node(mode(y))
        if self.max_depth == depth:
            return make_node(mode(y))
        if self.min_samples_split >= len(X):
            return make_node(mode(y))

        info_gains = [
            self._compute_information_gain(X, y, feature_idx)
            for feature_idx in range(X.shape[1])
        ]

        split_feature_idx = np.argmax(info_gains)

        # NOTE: `min_gain` hyperparameter is default to -1 bc the domain is [0, inf]
        if self.min_gain >= info_gains[split_feature_idx]:
            return make_node(mode(y))

        split_feature = self.feature_names_[split_feature_idx]
        split_node = make_node(split_feature, deep=True)

        if is_numeric_dtype(X[:, split_feature_idx]):
            split_node.threshold = self.possible_thresholds_[split_feature_idx]

        if DEBUG:
            print(f"\nSplit on Feature: {split_feature}")
            print(50 * "-")
            print(X)

        levels = [
            self._partition_data(X, y, split_feature_idx, level, split_node)
            for level in self.unique_levels_[split_feature_idx]
        ]

        if is_numeric_dtype(X[:, split_feature_idx]):
            levels = levels[0]

        for *partition_data, level in levels:
            split_node.add(
                self._tree_builder(
                    *partition_data, parent=split_node, branch=level, depth=depth + 1
                )
            )

        return split_node

    def predict_proba(self, X):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

        Returns
        -------
        np.ndarray
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        return np.apply_along_axis(lambda x: self._tree_walk(x).proba, axis=1, arr=X)

    def score(self, X, y):
        """Evaluates the performance of a decision tree classifier.

        The metric used to evaluate the performance of a decision tree
        classifier is `sklearn.metrics.accuracy_score`.

        Parameters
        ----------
        X : np.ndarray
            2D test feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

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


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    """Short Summary

    Extended Summary
    """

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
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of numerical values.

        Returns
        -------
        DecisionTreeRegressor
        """
        X, y = check_X_y(X, y, y_numeric=True)
        return self

    def _tree_builder(self, X, y, /, *, parent=None, branch=None, depth=0):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of numerical values.

        parent : DecisionTreeRegressor, default=None

        branch : str, default=None

        depth : int, default=0

        Returns
        -------
        DecisionNode
        """
        raise NotImplementedError

    def score(self, X, y):
        """Evaluates the performance of a decision tree regressor.

        The metric used to evaluate the performance of a decision tree
        regressor is `sklearn.metrics.mean_square_error`.

        Parameters
        ----------
        X : np.ndarray
            2D test feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

        y : np.ndarray
            1D test target array with shape (n_samples,) of numerical values.

        Returns
        -------
        float
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y, dtype=object, y_numeric=True)

        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred, squared=False)


class ParallelDecisionTreeClassifier(DecisionTreeClassifier):
    """Short Summary

    Extended Summary
    """

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

    def _tree_builder(
        self, X, y, /, comm=WORLD_COMM, *, parent=None, branch=None, depth=0
    ):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of
            categorical and/or numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of numerical values.

        comm : MPI.Intracomm, default=WORLD_COMM

        parent : DecisionTreeRegressor, default=None

        branch : str, default=None

        depth : int, default=0

        Returns
        -------
        DecisionNode
        """
        raise NotImplementedError
