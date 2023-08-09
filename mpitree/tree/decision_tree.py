""""""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from statistics import mode

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._node import DecisionNode


class BaseDecisionTree(metaclass=ABCMeta):
    """Short Summary

    Extended Summary

    Parameters
    ----------
    max_depth : int, optional
        A hyperparameter that upper bounds the number of splits.

    min_samples_split : int
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
        X : array-like
            2D test feature matrix with shape (n_samples, n_features) of numerical values.

        Returns
        -------
        ndarray
            An array of predicated leaf decision nodes.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        def tree_walk(query) -> DecisionNode:
            """Traverse a decision tree estimator from root to some leaf node.

            Decision nodes are queried based on the respective feature value
            from `x` at each level until some leaf decision node is reached.

            Parameters
            ----------
            x : array-like
                1D test instance array of shape (1, n_features).

            Returns
            -------
            DecisionNode
            """
            tree_node = self.tree_

            while not tree_node.is_leaf:
                if query[tree_node.value] <= tree_node.threshold:
                    tree_node = tree_node.children["<="]
                else:
                    tree_node = tree_node.children[">"]

            return tree_node

        return np.apply_along_axis(tree_walk, axis=1, arr=X)

    def predict(self, X):
        """Return the predicated leaf decision node.

        Extended Summary

        Parameters
        ----------
        X : array-like
            2D test feature matrix with shape (n_samples, n_features) of
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
            2D feature matrix with shape (n_samples, n_features) of numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        Returns
        -------
        DecisionTreeClassifier
        """
        X, y = check_X_y(X, y, dtype=object)

        self.classes_ = np.unique(y)
        self.n_thresholds_ = dict()

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
            2D feature matrix with shape (n_samples, n_features) of numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        feature_idx : int

        Returns
        -------
        max_gain : The maximum information gain value.
        """

        order = np.argsort(X[:, feature_idx])
        X_sorted, y_sorted = X[order], y[order]

        thresholds = [
            np.mean(X_sorted[i : i + 2, feature_idx])
            for i in np.where(np.diff(y_sorted) != 0)[0]
        ]

        def cond_entropy(t):
            mask = X[:, feature_idx] <= t
            regions = X[mask], X[~mask]
            weights = np.array([len(r) / len(X) for r in regions])
            impurity = np.array([self._entropy(y[mask]), self._entropy(y[~mask])])
            return weights @ impurity

        costs = [cond_entropy(t) for t in thresholds]

        self.n_thresholds_[feature_idx] = thresholds[np.argmin(costs)]
        max_gain = self._entropy(y) - min(costs)

        return max_gain

    def _make_tree(self, X, y, *, parent=None, branch=None, depth=0):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : array-like
            2D feature matrix with shape (n_samples, n_features) of numerical values.

        y : array-like
            1D target array with shape (n_samples,) of categorical values.

        parent : DecisionTreeClassifier, default=None

        branch : str, default=None

        depth : int, default=0

        Returns
        -------
        DecisionNode
        """

        def make_node(value, threshold=None):
            return deepcopy(
                DecisionNode(
                    value=value,
                    threshold=threshold,
                    branch=branch,
                    parent=parent,
                    target=y,
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

        split_feature = np.argmax(feature_importances)
        split_node = make_node(
            value=split_feature, threshold=self.n_thresholds_[split_feature]
        )

        mask = X[:, split_feature] <= split_node.threshold

        left_region, right_region = [X[mask], y[mask]], [X[~mask], y[~mask]]

        assert all(i.size for i in left_region)
        assert all(i.size for i in right_region)

        n_subtrees = [left_region + ["<="], right_region + [">"]]

        for X, y, branch in n_subtrees:
            subtree = self._make_tree(
                X,
                y,
                parent=split_node,
                branch=branch,
                depth=depth + 1,
            )
            split_node.children[branch] = subtree

        return split_node

    def score(self, X, y):
        """Evaluate the performance of a decision tree classifier.

        The metric used to evaluate the performance of a decision tree
        classifier is `sklearn.metrics.accuracy_score`.

        Parameters
        ----------
        X : array-like
            2D test feature matrix with shape (n_samples, n_features) of numerical values.

        y : array-like
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

    def _make_tree(self, X, y, *, parent=None, branch=None, depth=0, comm=WORLD_COMM):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        X : array-like
            2D feature matrix with shape (n_samples, n_features) of numerical values.

        y : array-like
            1D target array with shape (n_samples,) of numerical values.

        parent : DecisionNode, default=None

        branch : str, default=None

        depth : int, default=0

        comm : MPI.Intracomm, default=WORLD_COMM

        Returns
        -------
        DecisionNode
        """

        def make_node(value, threshold=None):
            return deepcopy(
                DecisionNode(
                    value=value,
                    threshold=threshold,
                    branch=branch,
                    parent=parent,
                    target=y,
                )
            )

        rank = comm.Get_rank()
        size = comm.Get_size()

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

        split_feature = np.argmax(feature_importances)
        split_node = make_node(
            value=split_feature, threshold=self.n_thresholds_[split_feature]
        )

        mask = X[:, split_feature] <= split_node.threshold

        left_region, right_region = [X[mask], y[mask]], [X[~mask], y[~mask]]

        assert all(i.size for i in left_region)
        assert all(i.size for i in right_region)

        n_subtrees = [left_region + ["<="], right_region + [">"]]

        if size == 1:
            for X, y, branch in n_subtrees:
                subtree = self._make_tree(
                    X,
                    y,
                    parent=split_node,
                    branch=branch,
                    depth=depth + 1,
                    comm=comm,
                )
                split_node.children[branch] = subtree
        else:
            group = self.Get_cyclic_dist(comm, n_block=2)

            level = rank % 2
            X, y, branch = n_subtrees[level]

            levels = comm.allgather(
                {
                    branch: self._make_tree(
                        X,
                        y,
                        parent=split_node,
                        branch=branch,
                        depth=depth + 1,
                        comm=group,
                    )
                }
            )

            for level in levels:
                for branch, tree_node in level.items():
                    split_node.children[branch] = tree_node

            group.Free()

        return split_node
