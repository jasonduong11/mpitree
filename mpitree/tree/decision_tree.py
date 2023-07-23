""""""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import deque
from copy import deepcopy
from statistics import mode

import graphviz
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._node import DecisionNode
from ._util import is_numeric_dtype

DEBUG = 0


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
    def __init__(self, *, max_depth, min_samples_split, estimator_type):
        self.max_depth = max_depth  # [0, inf)
        self.min_samples_split = min_samples_split  # [2, inf)
        DecisionNode._estimator_type = estimator_type

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
                tree_node = tree_node[query_level]
            return tree_node

        return np.apply_along_axis(tree_walk, axis=1, arr=X)

    @abstractmethod
    def _tree_builder(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: list,
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

        return np.vectorize(lambda node: node.feature)(self._decision_paths(X))

    def export_graphviz(self):
        """Short Summary

        Extended Summary

        Returns
        -------
        """

        check_is_fitted(self)

        def bfs(source):
            queue = deque([source])
            while queue:
                node = queue.popleft()
                children = node.children.values()
                yield node, children
                queue.extend(children)

        def info(node):
            return "\n".join(
                [
                    f"feature={node.feature}",
                    f"threshold={node.threshold}",
                    f"branch={node.branch}",
                    f"depth={node.depth}",
                    f"value={node.value}={node.n_samples}",
                    f"is_leaf={node.is_leaf}",
                ]
            )

        net = graphviz.Digraph(
            graph_attr={"size": "(6,6)"}, node_attr={"shape": "record"}
        )

        for node, children in bfs(self.tree_):
            net.node(info(node))
            for i, child in enumerate(children):
                net.node(info(child))
                net.edge(info(node), info(child), label=list(node.children)[i])

        return net


class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    def __init__(self, *, max_depth=None, min_samples_split=2):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            estimator_type="classifier",
        )

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
        if DEBUG:
            proba = np.proba(x)
            print(
                f"\t = H({set(np.unique(x))})",
                f"- {proba.round(3)} * log({proba.round(3)})",
                f"[{np.sum(proba * np.log2(proba)).round(3)}]",
                sep="\n\t\t = ",
            )
            return -np.sum(proba * np.log2(proba))

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

        def cost(x):
            threshold = x[-1]
            mask = X[:, feature_idx] < threshold
            levels = np.split_mask(X, mask)
            weights = np.array([len(level) / len(X) for level in levels])
            impurity = np.array([self._entropy(y[mask]), self._entropy(y[~mask])])
            if DEBUG:
                print(
                    f"{weights.round(3)} * {impurity.round(3)}",
                    np.dot(weights, impurity).round(3),
                    sep="\n\t = ",
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
            2D feature matrix with shape (n_samples, n_features) of either
            or both categorical and numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of categorical values.

        feature_idx : int

        Returns
        -------
        max_gain : The maximum information gain value.
        """
        if is_numeric_dtype(X[:, feature_idx]):
            (
                max_gain,
                self.n_thresholds_[feature_idx],
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
                    entropy = self._entropy(y[X[:, feature_idx] == level])
                    impurity.append(entropy)

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
            2D feature matrix with shape (n_samples, n_features) of either
            or both categorical and numerical values.

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

            assert all(i.size for i in np.split_mask(X, mask))
            assert all(i.size for i in np.split_mask(y, mask))

            n_subtrees = zip(
                np.split_mask(X, mask), np.split_mask(y, mask), ("True", "False")
            )
            return n_subtrees

        mask = X[:, feature_idx] == level

        if DEBUG:
            D = np.delete(X[mask], feature_idx, axis=1)
            print(50 * "-")
            print(np.column_stack((D, y[mask])))
            print("+")
            print(X[mask])
            return D, y[mask], level

        return np.delete(X[mask], feature_idx, axis=1), y[mask], level

    def _tree_builder(
        self, X, y, *, feature_indices=None, parent=None, branch=None, depth=0
    ):
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

        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))

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

        feature_importances = [
            self._compute_information_gain(X, y, feature_idx)
            for feature_idx in range(X.shape[1])
        ]

        split_feature_idx = np.argmax(feature_importances)
        norm_split_feature_idx = feature_indices[split_feature_idx]

        split_feature = self.feature_names_[norm_split_feature_idx]
        split_node = make_node(split_feature)

        if is_numeric_dtype(X[:, split_feature_idx]):
            split_node.threshold = self.n_thresholds_[split_feature_idx]

        if DEBUG:
            print(f"\nSplit on Feature: {split_feature}")
            print(50 * "-")
            print(X)

        n_subtrees = [
            self._partition_data(X, y, split_feature_idx, level, split_node)
            for level in self.unique_levels_[split_feature_idx]
        ]

        if is_numeric_dtype(X[:, split_feature_idx]):
            n_subtrees = n_subtrees[0]
        else:
            # since we remove a column on categorical features, we need to
            # keep track of the actual feature index Ex: [0, 1, 2] ->
            # remove categorical feature idx (0) -> [1, 2] So if the next
            # node is an categorical internal node and the split feature
            # idx of the partitioned dataset is (1), the corresponding
            # feature should be at index (2) from [1, 2] not (1) from [0,
            # 1]. NOTE: `feature_indices` should be equal for decision
            # nodes at the same level This does not occur if the decision
            # node is a leaf since the `feature` attribute would be the
            # target value and if the split categorical feature is at the
            # end ([0, 1, 2] -> [0, 1]) since the indices are preserved.
            del feature_indices[split_feature_idx]

        for X, y, branch in n_subtrees:
            subtree = self._tree_builder(
                X=X,
                y=y,
                feature_indices=deepcopy(feature_indices),
                parent=split_node,
                branch=branch,
                depth=depth + 1,
            )
            split_node.children[branch] = subtree

        del feature_indices
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


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    """Short Summary

    Extended Summary
    """

    def __init__(self, *, max_depth=np.inf, min_samples_split=1):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            estimator_type="regressor",
        )

    def fit(self, X, y):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of either
            or both categorical and numerical values.

        y : np.ndarray
            1D target array with shape (n_samples,) of numerical values.

        Returns
        -------
        DecisionTreeRegressor
        """
        X, y = check_X_y(X, y, y_numeric=True)
        return self

    def _tree_builder(self, *, X, y, parent=None, branch=None, depth=0):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : np.ndarray
            2D feature matrix with shape (n_samples, n_features) of either
            or both categorical and numerical values.

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
        """Evaluate the performance of a decision tree regressor.

        The metric used to evaluate the performance of a decision tree
        regressor is `sklearn.metrics.mean_square_error`.

        Parameters
        ----------
        X : np.ndarray
            2D test feature matrix with shape (n_samples, n_features) of
            either or both categorical and numerical values.

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

#     def _tree_builder(
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