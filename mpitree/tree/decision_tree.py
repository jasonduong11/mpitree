"""
This module defines the base decision tree, decision tree classifier, and
parallel decision tree classifier classes.
"""

from __future__ import annotations

from typing import override

from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._node import Node


class BaseDecisionTree:
    """A base decision tree class.

    Parameters
    ----------
    max_depth : int, optional
        A hyperparameter that upper bounds the number of splits per level.
        (the default is `None`, which implies the decision tree can be at
        any depth).

    min_samples_split : int, optional
        A hyperparameter that lower bounds the number of samples required
        to split. (the default is 2, which implies decision tree estimator
        may contain singleton nodes).
    """

    def __init__(self, *, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def __str__(self):
        """Return a text-based visualization of a decision tree estimator.

        The function is a wrapper function for the overloaded `iter`
        method that displays each string-formatted `Node` in a
        decision tree estimator.

        See Also
        --------
        mpitree.tree._node.__str__()

        Returns
        -------
        str
        """
        check_is_fitted(self)
        return "\n".join(map(str, self))

    def __iter__(self):
        """Perform a depth-first search on a decision tree estimator.

        The frontier is ordered by interior nodes first when nodes are
        pushed onto the stack. This provides the minimum number of
        disjoint branch components at each level.

        Yields
        ------
        Node
        """
        check_is_fitted(self)

        stack = [self.tree_]
        while stack:
            tree_node = stack.pop()
            yield tree_node

            stack.extend(
                sorted(tree_node.get_children(), key=lambda node: not node.is_leaf)
            )

    def _decision_paths(self, X):
        """Return a list of predicted leaf nodes.

        Parameters
        ----------
        X : array-like
            2D test feature array with shape (n_samples, n_features) of
            numerical values.

        Returns
        -------
        ndarray
            An array of predicated leaf decision nodes.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        def tree_walk(x) -> Node:
            """Traverse a decision tree from the root to some leaf node.

            Nodes are queried based on the respective feature value from
            `x` at each level until some leaf node has reached.

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
        """Return a predicted leaf node.

        The function is a wrapper function for the `_decision_paths`
        function that returns the `value` for each predicted leaf node.

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

        return np.vectorize(lambda node: node.value)(self._decision_paths(X))


class DecisionTreeClassifier(BaseDecisionTree, BaseEstimator, ClassifierMixin):
    """
    A decision tree classifier class.
    """

    def __init__(self, *, max_depth=None, min_samples_split=2):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)

    def fit(self, X, y):
        """Train a decision tree classifier.

        Parameters
        ----------
        X : array-like
            2D feature array with shape (n_samples, n_features) of
            numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        Returns
        -------
        DecisionTreeClassifier
        """
        X, y = check_X_y(X, y, dtype=object)

        self.n_features_ = X.shape[1]

        self.tree_ = self._make_tree(X, y)
        return self

    def _compute_entropy(self, x):
        """Measure the impurity on an array.

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
        """Return the difference of entropies before and after splitting.

        Parameters
        ----------
        X : array-like
            2D feature array with shape (n_samples, n_features) of
            numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        feature_idx : int
            The feature index value.

        Returns
        -------
        tuple
            The maximum information gain and optimal threshold value.
        """
        possible_thresholds = np.unique(X[:, feature_idx])

        costs = []
        for threshold in possible_thresholds:
            region = X[:, feature_idx] <= threshold
            levels = X[region], X[~region]
            weights = np.array([len(level) / len(X) for level in levels])
            impurity = np.array(
                [
                    self._compute_entropy(y[region]),
                    self._compute_entropy(y[~region]),
                ]
            )
            costs.append(np.dot(weights, impurity))

        return (
            self._compute_entropy(y) - min(costs),
            possible_thresholds[np.argmin(costs)],
        )

    def _make_tree(self, X, y, *, parent=None, depth=0):
        """Recursively constructs a decision tree classifier.

        Parameters
        ----------
        X : array-like
            2D feature array with shape (n_samples, n_features) of
            numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        parent : DecisionTreeClassifier, default=None
            The precedent node.

        depth : int, default=0
            The number of edges from the root node.

        Returns
        -------
        Node
        """
        n_classes = len(np.unique(y))
        n_samples = len(X)

        if (
            n_classes == 1
            or np.all(X == X[0])
            or (self.max_depth is not None and self.max_depth == depth)
            or n_samples < self.min_samples_split
        ):
            return Node(value=np.bincount(y).argmax(), parent=parent)

        info_gains, level_thresholds = zip(
            *[self._compute_information_gain(X, y, i) for i in range(self.n_features_)]
        )

        split_feature_idx = np.argmax(info_gains)
        split_threshold = level_thresholds[split_feature_idx]

        split_node = Node(
            value=split_feature_idx, threshold=split_threshold, parent=parent
        )

        region = X[:, split_feature_idx] <= split_threshold

        split_node.left = self._make_tree(
            X[region],
            y[region],
            parent=split_node,
            depth=depth + 1,
        )

        split_node.right = self._make_tree(
            X[~region],
            y[~region],
            parent=split_node,
            depth=depth + 1,
        )

        return split_node


class ParallelDecisionTreeClassifier(DecisionTreeClassifier):
    """
    A parallel decision classifier class.
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
            The current communicator.

        n_block : int, default=1
            The number of distributions.

        Returns
        -------
        MPI.Intracomm
        """
        rank = comm.Get_rank()
        key, color = divmod(rank, n_blocks)
        return comm.Split(color, key)

    @override
    def _make_tree(self, X, y, *, parent=None, depth=0, comm=WORLD_COMM):
        """Recursively constructs a parallel decision tree estimator.

        Parameters
        ----------
        X : array-like
            2D feature array with shape (n_samples, n_features)
            of numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        parent : Node, default=None
            The precedent node.

        depth : int, default=0
            The number of edges from the root node.

        comm : MPI.Intracomm, default=WORLD_COMM
            The current communicator.

        Returns
        -------
        Node
        """

        n_classes = len(np.unique(y))
        n_samples = len(X)

        rank = comm.Get_rank()
        size = comm.Get_size()

        if (
            n_classes == 1
            or np.all(X == X[0])
            or self.max_depth is not None
            and self.max_depth == depth
            or n_samples < self.min_samples_split
        ):
            return Node(value=np.bincount(y).argmax(), parent=parent)

        info_gains, level_thresholds = zip(
            *[self._compute_information_gain(X, y, i) for i in range(self.n_features_)]
        )

        split_feature_idx = np.argmax(info_gains)
        split_threshold = level_thresholds[split_feature_idx]

        split_node = Node(
            value=split_feature_idx, threshold=split_threshold, parent=parent
        )

        region = X[:, split_feature_idx] <= split_threshold

        if size == 1:
            split_node.left = self._make_tree(
                X[region],
                y[region],
                parent=split_node,
                depth=depth + 1,
                comm=comm,
            )

            split_node.right = self._make_tree(
                X[~region],
                y[~region],
                parent=split_node,
                depth=depth + 1,
                comm=comm,
            )
        else:
            group = self._get_cyclic_dist(comm, n_blocks=2)

            if rank % 2 == 0:
                X, y = X[region], y[region]
                level = "<="
            else:
                X, y = X[~region], y[~region]
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
