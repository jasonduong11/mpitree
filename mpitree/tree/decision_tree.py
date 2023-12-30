""""""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from statistics import mode

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._node import Node


class BaseDecisionTree(metaclass=ABCMeta):
    """Short Summary

    Extended Summary

    Parameters
    ----------
    max_depth : int, optional
        A hyperparameter that upper bounds the number of splits.

    min_samples_split : int, optional
        A hyperparameter that lower bounds the number of samples required to
        split.
    """

    @abstractmethod
    def __init__(self, *, max_depth: int, min_samples_split: int):
        self.max_depth = max_depth  # [0, inf)
        self.min_samples_split = min_samples_split  # [2, inf)

    def __str__(self):
        """Export a text-based visualization of a decision tree estimator.

        The function is a wrapper function for the overloaded `iter`
        method that displays each string-formatted `Node` in a
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
        Node

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

            if not tree_node.is_leaf:
                frontier.extend(
                    sorted(
                        [tree_node.left, tree_node.right], key=lambda n: not n.is_leaf
                    )
                )

    def _decision_paths(self, X):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : array-like
            2D test feature array with shape (n_samples, n_features) of numerical values.

        Returns
        -------
        ndarray
            An array of predicated leaf decision nodes.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        def tree_walk(x) -> Node:
            """Traverse a decision tree estimator from root to some leaf node.

            Decision nodes are queried based on the respective feature value
            from `x` at each level until some leaf decision node is reached.

            Parameters
            ----------
            x : array-like
                1D test instance array of shape (1, n_features).

            Returns
            -------
            Node
            """
            tree_node = self.tree_

            while not tree_node.is_leaf:
                if x[tree_node.value] <= tree_node.threshold:
                    tree_node = tree_node.left
                else:
                    tree_node = tree_node.right

            return tree_node

        return np.apply_along_axis(tree_walk, axis=1, arr=X)

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

        return np.vectorize(lambda n: n.value)(self._decision_paths(X))


class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, *, max_depth=None, min_samples_split=2):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)

    def fit(self, X, y):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : array-like
            2D feature array with shape (n_samples, n_features) of numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        Returns
        -------
        DecisionTreeClassifier
        """
        X, y = check_X_y(X, y, dtype=object)

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        self.tree_ = self._make_tree(X, y)
        return self

    def _entropy(self, x):
        """Measure the impurity on some feature.

        The parameter `x` is assumed to represent values for some
        particular feature (including the label or target feature) on some
        dataset `X`.

        Parameters
        ----------
        x : array-like
            1D feature column array of shape (n_samples,).

        Returns
        -------
        float
        """

        _, n_unique_classes = np.unique(x, return_counts=True)
        proba = n_unique_classes / len(x)
        return -np.sum(proba * np.log2(proba))

    def _compute_information_gain(self, X, y, feature_idx):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : array-like
            2D feature array with shape (n_samples, n_features) of numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        feature_idx : int

        Returns
        -------
        max_gain : The maximum information gain value.
        """
        possible_thresholds = np.unique(X[:, feature_idx])

        def cond_entropy(t):
            region_bound = X[:, feature_idx] <= t
            levels = X[region_bound], X[~region_bound]
            weights = np.array([len(level) / len(X) for level in levels])
            impurity = np.array(
                [self._entropy(y[region_bound]), self._entropy(y[~region_bound])]
            )
            return weights @ impurity

        costs = [cond_entropy(t) for t in possible_thresholds]

        return (
            self._entropy(y) - min(costs),
            possible_thresholds[np.argmin(costs)],
        )

    def _make_tree(self, X, y, *, parent=None, depth=0):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : array-like
            2D feature array with shape (n_samples, n_features) of numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        parent : DecisionTreeClassifier, default=None

        level: str, default=None

        depth : int, default=0

        Returns
        -------
        Node
        """
        n_classes = len(np.unique(y))
        n_samples = len(X)

        def make_node(value, threshold=None):
            return Node(
                value=value,
                threshold=threshold,
                parent=parent,
            )

        if (
            n_classes == 1
            or np.all(X == X[0])
            or (self.max_depth is not None and self.max_depth == depth)
            or n_samples < self.min_samples_split
        ):
            return make_node(mode(y))

        info_gains, level_thresholds = zip(
            *[self._compute_information_gain(X, y, i) for i in range(self.n_features_)]
        )

        split_feature_idx = np.argmax(info_gains)
        split_threshold = level_thresholds[split_feature_idx]

        split_node = make_node(value=split_feature_idx, threshold=split_threshold)

        region_mask = X[:, split_feature_idx] <= split_threshold

        split_node.left = self._make_tree(
            X[region_mask],
            y[region_mask],
            parent=split_node,
            depth=depth + 1,
        )

        split_node.right = self._make_tree(
            X[~region_mask],
            y[~region_mask],
            parent=split_node,
            depth=depth + 1,
        )

        return split_node

    def score(self, X, y):
        """Evaluate the performance of a decision tree classifier.

        The metric used to evaluate the performance of a decision tree
        classifier is the accuracy score.

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


class ParallelDecisionTreeClassifier(DecisionTreeClassifier):
    """Short Summary

    Extended Summary
    """

    from mpi4py import MPI

    WORLD_COMM = MPI.COMM_WORLD
    WORLD_RANK = WORLD_COMM.Get_rank()
    WORLD_SIZE = WORLD_COMM.Get_size()

    def _get_cyclic_dist(
        self, comm: MPI.Intracomm = None, *, n_blocks: int = 1
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
        key, color = divmod(rank, n_blocks)
        return comm.Split(color, key)

    def _make_tree(self, X, y, *, parent=None, depth=0, comm=WORLD_COMM):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : array-like
            2D feature array with shape (n_samples, n_features) of numerical values.

        y : array-like
            1D target array with shape (n_samples,) of numerical values.

        parent : Node, default=None

        level: str, default=None

        depth : int, default=0

        comm : MPI.Intracomm, default=WORLD_COMM

        Returns
        -------
        Node
        """

        n_classes = len(np.unique(y))
        n_samples = len(X)

        rank = comm.Get_rank()
        size = comm.Get_size()

        def make_node(value, threshold=None):
            return Node(
                value=value,
                threshold=threshold,
                parent=parent,
            )

        if (
            n_classes == 1
            or np.all(X == X[0])
            or self.max_depth is not None
            and self.max_depth == depth
            or n_samples < self.min_samples_split
        ):
            return make_node(mode(y))

        info_gains, level_thresholds = zip(
            *[self._compute_information_gain(X, y, i) for i in range(self.n_features_)]
        )

        split_feature_idx = np.argmax(info_gains)
        split_threshold = level_thresholds[split_feature_idx]

        split_node = make_node(value=split_feature_idx, threshold=split_threshold)

        region_mask = X[:, split_feature_idx] <= split_threshold

        if size == 1:
            split_node.left = self._make_tree(
                X[region_mask],
                y[region_mask],
                parent=split_node,
                depth=depth + 1,
                comm=comm,
            )

            split_node.right = self._make_tree(
                X[~region_mask],
                y[~region_mask],
                parent=split_node,
                depth=depth + 1,
                comm=comm,
            )
        else:
            group = self._get_cyclic_dist(comm, n_blocks=2)

            if rank % 2 == 0:
                X, y = X[region_mask], y[region_mask]
                level = "<="
            else:
                X, y = X[~region_mask], y[~region_mask]
                level = ">"

            levels = comm.allgather(
                {
                    level: self._make_tree(
                        X,
                        y,
                        parent=split_node,
                        depth=depth + 1,
                        comm=group,
                    )
                }
            )

            for level in levels:
                for sign, subtree in level.items():
                    if sign == "<=":
                        split_node.left = subtree
                    elif sign == ">":
                        split_node.right = subtree

            group.Free()

        return split_node
