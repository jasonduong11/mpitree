"""This module contains the decision tree base estimator."""

from __future__ import annotations

import abc
from itertools import pairwise, starmap
from operator import eq, ge, lt

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from tabulate import tabulate

from ._node import DecisionNode, logger


class DecisionTreeEstimator(metaclass=abc.ABCMeta):
    """A decision tree estimator.

    A decision tree base class for decision tree implementations specific
    to some learning task (e.g., classification and regression).

    Parameters
    ----------
    root : Node, default=None
        The start node with depth zero of a decision tree.

    purity : {'find_entropy', 'find_variance'}
        The measure of impurity used for calculating the information gain.

    criterion : {'max_depth', 'min_samples_split', 'min_gain'}, optional
        Contains pre-pruning hyperparameters of a decision tree (the
        default is None and will be assigned as an empty dictionary).

    References
    ----------
    .. [1] J. D. Kelleher, M. B. Namee, and A. D'Arcy, Fundamentals
        of machine learning for Predictive Data Analytics: Algorithms,
        worked examples, and case studies. Cambridge, MA: The MIT
        Press, 2020.
    """

    _validate_parameter = {
        "criterion": {"max_depth", "min_samples_split", "min_gain"},
        "splitter": {"ID3", "CART"},
        "purity": {"entropy", "gini"},
    }
    _root = None
    _n_levels = None
    _n_thresholds = {}

    @abc.abstractmethod
    def __init__(
        self,
        *,
        criterion=None,
        splitter="ID3",
        purity="entropy",
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.purity = purity

    def __iter__(self, other=None):
        """Perform a depth-first search on the decision tree.

        The traversal starts at the root node and recursively traverses
        across all its children in a fixed-order. The `values` method
        assures children nodes are searched in a stack-like manner.

        Parameters
        ----------
        other : DecisionNode, optional
            The subsequent node of the depth-first traversal.

        Yields
        ------
        DecisionNode

        See Also
        --------
        DecisionTreeEstimator.__repr__ :
            Return a string-formatted decision tree.
        """
        if not other:
            other = self._root

        yield other
        for child_node in other.children.values():
            yield from self.__iter__(child_node)

    def __repr__(self):
        """Return a string-formatted decision tree.

        The output string of a decision tree is delimited by a single new-
        line escape character before string-formatting every `Node`
        object.

        Returns
        -------
        str
            The string-formatted decision tree.

        Raises
        ------
        AttributeError
            If the decision tree has not been fitted.

        See Also
        --------
        DecisionTreeEstimator.__iter__ :
            Perform a depth-first search on the decision tree.
        """
        if not self.check_is_fitted:
            raise AttributeError("Decision tree is not fitted")
        return "\n".join(map(str, self))

    def __eq__(self, other: DecisionTreeEstimator):
        """Check if two decision trees are equivalent.

        Performs a pair-wise comparison among attributes of both
        `DecisionTreeEstimator` objects and returns true if all attributes
        are equal; otherwise, returns false. The function will raise a
        TypeError if an instance is not a `DecisionTreeEstimator` object.
        If either comparison object is not fitted, the function will raise
        an AttributeError.

        Parameters
        ----------
        other : DecisionTreeEstimator
            The comparision object.

        Returns
        -------
        bool
            Returns true if both `DecisionTreeEstimator` objects contain
            identical values for all attributes.

        Raises
        ------
        TypeError
            If `other` is not type `DecisionTreeEstimator`.
        AttributeError
            If either comparison objects are not fitted.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                "Expected type `DecisionTreeEstimator` but got type `%s`", type(other)
            )
        if not self.check_is_fitted or not other.check_is_fitted:
            raise AttributeError("At least one 'DecisionTreeEstimator' is not fitted")

        return all(starmap(eq, zip(self, other)))

    @property
    def n_levels(self):
        """ """
        return self._n_levels

    @property
    def n_thresholds(self):
        """ """
        return self._n_thresholds

    @property
    def check_is_fitted(self):
        """Check whether a decision tree is fitted.

        The decision tree is fitted if it calls the `fit` method where the
        resulting call will instantiate the root as a `Node` object.

        Returns
        -------
        bool
            Returns true if the root node is a `Node` object.
        """
        return isinstance(self._root, DecisionNode)

    def _check_valid_params(self, X, y, /):
        """Check parameters have valid values.

        The input data must not be empty. Both thresholds, and criterion
        attributes must be `dict` types and the criterion attribute must
        have valid keys.

        Parameters
        ----------
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.

        Returns
        -------
        None
            Returns None when parameters have valid values.

        Raises
        ------
        Exception
            If either datasets `X` or `y` are empty.
        TypeError
            If `criterion` is not a `dict` types.
        KeyError
            If any unexpected hyperparameters keys.
        ValueError
            If any `criterion` values are negative.

        See Also
        --------
        decision_tree.DecisionTreeClassifier.fit :
            Performs the ID3 (Iterative Dichotomiser 3) algorithm.

        Examples
        --------
        Decision tree with one invalid key for the criterion attribute:

        >>> import pandas as pd
        >>> from mpitree.decision_tree import DecisionTreeClassifier
        >>> df = pd.DataFrame({"A": ["a"], "y": ["+"]})
        >>> X, y = df.iloc[:, :-1], df.iloc[:, -1]
        >>> DecisionTreeClassifier(
        ...     criterion={"fake_key": 1, "max_depth": 2}
        ... )._check_valid_params(X, y)
        Traceback (most recent call last):
            ...
        KeyError: "Found some unexpected keys: {'fake_key'}"
        """
        if X.empty or y.empty:
            raise Exception("Expected at least one sample in both X and y")

        if self.criterion is None:
            self.criterion = {}
        elif not isinstance(self.criterion, dict):
            raise TypeError(
                "Expected `criterion` attribute to be type `dict` but got type `%s`"
                % type(self.criterion)
            )
        elif keyerr := set(self.criterion) - self._validate_parameter["criterion"]:
            raise KeyError("Found some unexpected keys: %s" % keyerr)
        elif any(i < 0 for i in self.criterion.values()):
            raise ValueError("Criterion values must be positive integers")

        self._n_levels = {
            d: ((lt, ge) if is_numeric_dtype(X[d]) else np.unique(X[d]))
            for d in X.columns
        }

    def find_optimal_threshold(self, X, y, d):
        """Compute the optimal threshold between different target levels.

        The optimal threshold is found by first sorting `X` with respect
        to feature `d`. A pair-wise comparison is applied to each con
        secutive pairs and the possible thresholds are the mean of the
        feature values of different target levels. For each possible
        threshold, the split providing the highest information gain is
        selected as the optimal threshold.

        Parameters
        ----------
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.
        d : str
            The feature specified in `X`.

        Returns
        -------
        tuple
            The max information gain, the optimal threshold, and the re
            maining uncertainty.
        """
        df = pd.concat([X, y], axis=1)
        df.sort_values(by=[d], inplace=True)

        logger.info("Sorted according to feature '%s'", d)
        logger.info(tabulate(df.head(), list(df), floatfmt=".3f", tablefmt="pretty"))

        y_sorted = df.iloc[:, -1]
        thresholds = []
        for i, j in zip(pairwise(y_sorted.index), pairwise(y_sorted.values)):
            if is_numeric_dtype(X[d]):
                thresholds.append(X.loc[pd.Index(i), d].mean())
            else:
                if j[0] != j[1]:
                    thresholds.append(X.loc[pd.Index(i), d].mean())

        logger.info(
            "Possible Thresholds (Rounded 2): %s",
            [round(t, 2) for t in thresholds],
        )

        levels = []
        for threshold in thresholds:
            level = df[df[d] < threshold], df[df[d] >= threshold]
            weight = np.array([len(i) / len(df) for i in level])

            metric_total = self.purity(df.iloc[:, :-1], df.iloc[:, -1])
            metric_partial = [
                self.purity(level[0].iloc[:, :-1], level[0].iloc[:, -1]),
                self.purity(level[1].iloc[:, :-1], level[1].iloc[:, -1]),
            ]

            rem = np.dot(weight, metric_partial)
            levels.append(metric_total - rem)

        t_hat = thresholds[np.argmax(levels)]

        if not X[d].min() <= t_hat <= X[d].max():
            logger.error("%f not in range [%f, %f]", t_hat, X[d].min(), X[d].max())

        return max(levels), t_hat, rem

    def partition_data(self, X, y, d, op, threshold=None):
        """Return a subset of the data for some feature and level.

        Parameters
        ----------
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.
        d : str
            The feature specified in `X`.
        """
        if threshold:
            return (
                *list(map(lambda f: f[op(X[d], threshold)], [X, y])),
                f"{'<' if op is lt else '>='} {threshold}",
            )
        idx = X[d] == op
        return X.loc[idx].drop(d, axis=1), y.loc[idx], op

    def predict(self, X):
        """Predict a test sample on a decision tree.

        The function traverses the decision tree by looking up the feature
        value at the current node from the test sample `x`. If the current
        feature contains numerical values, the left transition is always
        considered if the `x` is less than the threshold; otherwise, it
        considers the right transition. If the feature contains cate
        gorical values, the feature value at `x` is used as a key to the
        current node children.

        Parameters
        ----------
        x : pd.DataFrame.dtypes
            The single test sample.

        Returns
        -------
        node.feature : str or float
            The class (classification) label or real (regression) value
            for a prediction on `x`.

        Raises
        ------
        KeyError
            If a feature does not satisfy any value specified in `x`.

        See Also
        --------
        decision_tree.DecisionTreeClasifier.score :
            Evaluate the decision tree model on the test set.
        """
        if not self.check_is_fitted:
            raise AttributeError("Decision tree is not fitted")

        node = self._root
        while not node.is_leaf:
            query_branch = X[node.feature]

            if is_numeric_dtype(query_branch):
                next_node = node.left if query_branch < node.threshold else node.right
            else:
                try:
                    next_node = node.children[query_branch]
                except KeyError:
                    logger.error(
                        "Branch %s -> %s does not exist",
                        node.feature,
                        query_branch,
                        exec_info=True,
                    )
            node = next_node
        return node.feature

    @abc.abstractmethod
    def fit(self, X, y, /):
        ...

    @abc.abstractmethod
    def predict_prob(self, X, /):
        ...

    @abc.abstractmethod
    def score(self, X, y, /):
        ...
