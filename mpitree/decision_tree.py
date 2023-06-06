"""This module contains various decision tree implementations.

It includes decision trees for both learning tasks, classification and
regression and a parallel version for classification.

"""

from copy import deepcopy
from multiprocessing.pool import ThreadPool
from statistics import mean, mode

import numpy as np
from mpi4py import MPI
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score, mean_squared_error

from ._base_estimator import DecisionTreeEstimator
from ._node import DecisionNode, logger
from ._utils.conversion import _to_pandas_dataframe

WORLD_COMM = MPI.COMM_WORLD
WORLD_RANK = WORLD_COMM.Get_rank()
WORLD_SIZE = WORLD_COMM.Get_size()


def Get_cyclic_dist(comm: MPI.Intracomm = None, *, n_block: int = 1) -> MPI.Intracomm:
    """Schedules processes in a cyclic distribution.

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


MPI.Get_cyclic_dist = Get_cyclic_dist


class DecisionTreeClassifier(DecisionTreeEstimator):
    """A decision tree classifier.

    A decision tree whose prediction is a label from a discrete set of
    classes.

    Parameters
    ----------
    criterion : {'max_depth', 'min_samples_split', 'min_gain'}, optional
        Contains pre-pruning hyperparameters of a decision tree (the
        default is None and will be assigned as an empty dictionary).

    Examples
    --------
    Decision tree classifier to predict vegetation distributions with
    categorical and numerical attributes:

    >>> import pandas as pd
    >>> from mpitree.decision_tree import DecisionTreeClassifier
    >>> data = pd.DataFrame(
    ...     {
    ...         "Stream": [
    ...             "false",
    ...             "true",
    ...             "true",
    ...             "false",
    ...             "false",
    ...             "true",
    ...             "true"
    ...         ],
    ...         "Slope": [
    ...             "steep",
    ...             "moderate",
    ...             "steep",
    ...             "steep",
    ...             "flat",
    ...             "steep",
    ...             "steep",
    ...         ],
    ...         "Elevation": [3900, 300, 1500, 1200, 4450, 5000, 3000],
    ...         "Vegetation": [
    ...             "chapparal",
    ...             "riparian",
    ...             "riparian",
    ...             "chapparal",
    ...             "conifer",
    ...             "conifer",
    ...             "chapparal",
    ...         ],
    ...     }
    ... )
    >>> X, y = data.iloc[:, :-1], data.iloc[:, -1]
    >>> clf = DecisionTreeClassifier().fit(X, y)
    >>> print(clf)
    ┌── Elevation
    │  ├── Stream [< 4175.0]
    │  │  └── chapparal [false]
    │  │  ├── Elevation [true]
    │  │  │  └── riparian [< 2250.0]
    │  │  │  └── chapparal [>= 2250.0]
    │  └── conifer [>= 4175.0]

    Decision tree classifier with hyperparameter `max_depth` = 3 for the
    ``load_iris`` dataset provided by *scikit-learn*:

    >>> import pandas as pd
    >>> from mpitree.decision_tree import DecisionTreeClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> iris = load_iris(as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     iris.data, iris.target, test_size=0.20, random_state=42
    ... )
    >>> clf = DecisionTreeClassifier(
    ...     criterion={"max_depth": 3}
    ... ).fit(X_train, y_train)
    >>> print(clf)
    ┌── petal length (cm)
    │  └── 0 [< 2.45]
    │  ├── petal length (cm) [>= 2.45]
    │  │  ├── petal width (cm) [< 4.75]
    │  │  │  └── 1 [< 1.65]
    │  │  │  └── 2 [>= 1.65]
    │  │  ├── petal width (cm) [>= 4.75]
    │  │  │  └── 2 [< 1.75]
    │  │  │  └── 2 [>= 1.75]
    >>> train_score = clf.score(X_train, y_train)
    >>> print(f"Train Accuracy: {train_score:.2%}")
    Train Accuracy: 95.83%
    >>> test_score = clf.score(X_test, y_test)
    >>> print(f"Test Accuracy: {test_score:.2%}")
    Test Accuracy: 96.67%
    """

    def __init__(self, *, criterion=None):
        super().__init__(purity=self.entropy, criterion=criterion)

    def entropy(self, X, y, /):
        """Measure the amount of impurity.

        Entropy is a weighted sum of the logs of the probabilities of se
        lecting a random value from a feature and is measured in bits. A
        higher value for entropy corresponds to a lower probability and
        vice versa.

        Parameters
        ----------
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.

        Returns
        -------
        float
        """
        proba = np.unique(y, return_counts=True)[1] / len(X)
        return -np.sum(proba * np.log2(proba))

    def cond_entropy(self, X, y, d, /):
        """Measure the resultant entropy on a tested feature.

        Conditional entropy is the entropy is measured on a partitioned dataset.

        Parameters
        ----------
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.
        d : str
            The feature specified in `X`.

        Returns
        -------
        float
        """
        weight = np.unique(X[d], return_counts=True)[1] / len(X)
        purity = [self.purity(X[X[d] == t], y[X[d] == t]) for t in np.unique(X[d])]
        return np.dot(weight, purity)

    def find_information_gain(self, X, y, d, /):
        """Measure the reduction in the overall entropy given a feature.

        Information gain is the measure of the reduction in the overall
        entropy from a set of instances achieved by testing on a given
        feature. A higher value for an information gain corresponds to
        a better split.

        Parameters
        ----------
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.
        d : str
            The feature specified in `X`.

        Returns
        -------
        float
        """
        if is_numeric_dtype(X[d]):
            gain, optimal_threshold, rem = super().find_optimal_threshold(X, y, d)
            self.n_thresholds[d] = optimal_threshold
            logger.info("%s = %d - %d = %d", d, self.purity(X, y), rem, gain)
        else:
            gain = self.purity(X, y) - self.cond_entropy(X, y, d)
            logger.info(
                "%s = %d - %d = %d",
                d,
                self.purity(X, y),
                self.cond_entropy(X, y, d),
                gain,
            )
        return gain

    def fit(self, X, y, /):
        """Fit the decision tree classifier to the dataset.

        The `fit` method induces a decision tree model through a greedy
        search in the hypothesis space.

        Parameters
        ----------
        X : pd.DataFrame.dtypes
            The feature matrix of the dataset.
        y : pd.Series.dtypes
            The target vector of the dataset.

        Returns
        -------
        self
            The current instance of the `DecisionTreeClassifier`.
        """
        X, y = _to_pandas_dataframe(X, y)
        self._check_valid_params(X, y)
        self._root = self._make_tree(X, y)
        return self

    def _make_tree(self, X, y, /, *, parent_y=None, branch=None, depth=0):
        """Perform the ID3 (Iterative Dichotomiser 3) algorithm.

        The ID3 algorithm recursively grows the tree in a depth-first
        search manner by finding the optimal feature to split and
        repeating until some base case is reached. If the root becomes a
        leaf node, its branch is labeled with the majority value for a
        random feature. The splitting criteria are based on information
        gain, and the standard approach to measure impurity is entropy.
        The feature that provides the highest information gain among the
        possible feature levels is chosen for a particular path. Each
        recursive call partitions the current dataset according to the
        feature values for their respective level.

        Parameters
        __________
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.

        parent_y : pandas.core.series.Series, default=None
            The parent target feature.

        branch : str, default=None
            The transition from which the parent parition.

        depth : int, default=0
            The current level at a node.

        Returns
        -------
        best_node : Node
            The root node of the decision tree classifier.

        Notes
        -----
        - All instances have the same labels.
        - Dataset is empty.
        - If all feature values are identical.
        - Max depth reached.
        - Max number of instances in partitioned dataset reached.
        """

        def make_node(value):
            return DecisionNode(
                feature=value, branch=branch, depth=depth, shape=dict(y.value_counts())
            )

        if len(np.unique(y)) == 1:
            logger.info("All instances have the same labels (%s)", mode(y))
            return make_node(mode(y))
        if X.empty:
            logger.info("Dataset is empty")
            return make_node(mode(parent_y))
        if all((X[d] == X[d].iloc[0]).all() for d in X.columns):
            logger.info("All instances have the same descriptive features")
            return make_node(mode(y))
        if self.criterion.get("max_depth", np.inf) <= depth:
            logger.info("Stopping at Max Depth: %d", depth)
            return make_node(mode(y))
        if self.criterion.get("min_samples_split", -np.inf) >= len(X):
            logger.info("Stopping at %d instances", len(X))
            return make_node(mode(y))

        logger.info("\n===Information Gain===\n")
        max_gain = np.argmax([self.find_information_gain(X, y, d) for d in X.columns])

        if self.criterion.get("min_gain", -np.inf) >= max_gain:
            logger.info("Stopping at Information Gain=%d", max_gain)
            return make_node(mode(y))

        best_feature = X.columns[max_gain]
        logger.info("\nBest Feature = %s", best_feature)

        best_node = deepcopy(make_node(best_feature))

        if is_numeric_dtype(X[best_feature]):
            best_node.threshold = self.n_thresholds[best_feature]
            logger.info("Optimal Threshold = %f", best_node.threshold)

        levels = [
            self.partition_data(X, y, best_feature, level, best_node.threshold)
            for level in self.n_levels[best_feature]
        ]

        for *partition_data, level in levels:
            best_node.add(
                self._make_tree(
                    *partition_data, parent_y=y, branch=level, depth=depth + 1
                )
            )
        return best_node

    def predict_prob(self, X, /):
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
        return np.array(list(map(lambda f: f / node.samples, node.shape)))

    def score(self, X, y, /):
        """Evaluate the decision tree model on the test set.

        The decision tree classifier makes predictions for each test
        sample and outputs the accuracy.

        Parameters
        ----------
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.

        Returns
        -------
        float
        """
        X, y = _to_pandas_dataframe(X, y)
        y_hat = [self.predict(X.iloc[i]) for i in range(len(X))]
        return accuracy_score(y, y_hat)


class DecisionTreeRegressor(DecisionTreeEstimator):
    """A decision tree regressor.

    A decision tree whose output is a target is from the set of real
    numbers.

    Parameters
    ----------
    criterion : {'max_depth', 'min_samples_split', 'min_gain'}, optional
        Contains pre-pruning hyperparameters of a decision tree (the
        default is None and will be assigned as an empty dictionary).

    Examples
    --------
    Decision tree regressor to predict bike rentals:

    >>> import pandas as pd
    >>> from mpitree.decision_tree import DecisionTreeRegressor
    >>> data = {
    ...     "Season": [
    ...         "winter",
    ...         "winter",
    ...         "winter",
    ...         "spring",
    ...         "spring",
    ...         "spring",
    ...         "summer",
    ...         "summer",
    ...         "summer",
    ...         "autumn",
    ...         "autumn",
    ...         "autumn",
    ...     ],
    ...     "Work Day": [
    ...         "false",
    ...         "false",
    ...         "true",
    ...         "false",
    ...         "true",
    ...         "true",
    ...         "false",
    ...         "true",
    ...         "true",
    ...         "false",
    ...         "false",
    ...         "true",
    ...     ],
    ...     "Rentals": [
    ...         800,
    ...         826,
    ...         900,
    ...         2100,
    ...         4740,
    ...         4900,
    ...         3000,
    ...         5800,
    ...         6200,
    ...         2910,
    ...         2880,
    ...         2820,
    ...     ],
    ... }
    >>> df = pd.DataFrame(data)
    >>> X, y = df.iloc[:, :-1], df.iloc[:, -1]
    >>> regr = DecisionTreeRegressor().fit(X, y)
    >>> print(regr)
    ┌── Season
    │  ├── Work Day [autumn]
    │  │  └── 2895 [false]
    │  │  └── 2820 [true]
    │  ├── Work Day [spring]
    │  │  └── 2100 [false]
    │  │  └── 4820 [true]
    │  ├── Work Day [summer]
    │  │  └── 3000 [false]
    │  │  └── 6000 [true]
    │  ├── Work Day [winter]
    │  │  └── 813 [false]
    │  │  └── 900 [true]
    """

    def __init__(self, *, criterion=None):
        super().__init__(purity=self.find_variance, criterion=criterion)

    def find_variance(self, X, y, /):
        """Compute the variance on each target level.

        The variance refers to the amount of dispersion for values in a
        given feature.

        Parameters
        ----------
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.

        Returns
        -------
        float
        """
        if len(X) == 1:
            logger.info("Variance = 0")
            return 0.0
        logger.info(
            "Variance = %f / %d = %f",
            np.sum([(t - mean(y)) ** 2 for t in y]),
            len(X) - 1,
            np.sum([(t - mean(y)) ** 2 for t in y]) / (len(X) - 1),
        )
        return np.sum([(t - mean(y)) ** 2 for t in y]) / (len(X) - 1)

    def find_weighted_variance(self, X, y, d, /):
        """Compute the weighted variance on each target level.

        The weighted variance computes the variance for each value of a
        given feature multiplied by the ratio between the partitioned and
        original dataset.

        Parameters
        ----------
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.
        d : str
            The feature specified in `X`.

        Returns
        -------
        float
        """
        if is_numeric_dtype(X[d]):
            gain, optimal_threshold, _ = super().find_optimal_threshold(X, y, d)
            self.n_thresholds[d] = optimal_threshold

            # # TODO: Could cause future bugs
            # # larger reduction in variance should count more
            # # small offset added with variance reduction is zero
            # # (aka, worst splitting)
            # gain = np.reciprocal(gain + 1e-4)
            logger.info(gain)
        else:
            weight = np.unique(X[d], return_counts=True)[1] / len(X)
            purity = [self.purity(X[X[d] == t], y[X[d] == t]) for t in np.unique(X[d])]
            gain = np.dot(weight, purity)
        return gain

    def fit(self, X, y, /):
        """Fit the decision tree regressor to the dataset.

        The `fit` method induces a decision tree model through a greedy
        search in the hypothesis space.

        Parameters
        ----------
        X : pd.DataFrame.dtypes
            The feature matrix of the dataset.
        y : pd.Series.dtypes
            The target vector of the dataset.
        d : str
            The feature specified in `X`.

        Returns
        -------
        self
            The current instance of the `DecisionTreeRegressor`.
        """
        X, y = _to_pandas_dataframe(X, y)
        self._check_valid_params(X, y)
        self._root = self._make_tree(X, y)
        return self

    def _make_tree(self, X, y, /, *, branch=None, depth=0):
        """Perform the ID3 algorithm.

        Parameters
        __________
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.

        branch : str, default=None
            The transition from which the parent parition.

        depth : int, default=0
            The current level at a node.

        Returns
        -------
        best_node : Node
            The root node of the decision tree regressor

        Notes
        -----
        - All instances have the labels.
        - Dataset is empty.
        - If all feature values are identical.
        - Max depth reached.
        - Min number of instances in partitioned dataset reached.
        """

        def make_node(value):
            return DecisionNode(feature=value, branch=branch, depth=depth)

        if len(np.unique(y)) == 1:
            logger.info("All instances have the same labels (%s)", str(y.mean()))
            # return make_node(mean(y))
            return make_node(y.mean())
        if X.empty:
            logger.info("Dataset is empty")
            # return make_node(mean(y))
            return make_node(y.mean())
        if all((X[d] == X[d].iloc[0]).all() for d in X.columns):
            logger.info("All instances have the same descriptive features")
            # return make_node(mean(y))
            return make_node(y.mean())
        if self.criterion.get("max_depth", np.inf) <= depth:
            logger.info("Stopping at Max Depth: %d", depth)
            # return make_node(mean(y))
            return make_node(y.mean())
        if self.criterion.get("min_samples_split", -np.inf) >= len(X):
            logger.info("Stopping at %d instances", len(X))
            # return make_node(mean(y))
            return make_node(y.mean())

        logger.info("\n===Information Gain===\n")

        var = np.argmin([self.find_weighted_variance(X, y, d) for d in X.columns])
        logger.info("Var = %d", var)

        best_feature = X.columns[var]
        logger.info("Best Feature = %s", best_feature)

        best_node = deepcopy(make_node(best_feature))

        if is_numeric_dtype(X[best_feature]):
            best_node.threshold = round(self.n_thresholds[best_feature], 2)
            logger.info("Possible Optimal Thresholds = %s", self.n_thresholds)
            logger.info("Optimal Threshold = %f", best_node.threshold)

        levels = [
            self.partition_data(X, y, best_feature, level, best_node.threshold)
            for level in self.n_levels[best_feature]
        ]

        for *partition_data, level in levels:
            best_node.add(
                self._make_tree(*partition_data, branch=level, depth=depth + 1)
            )
        return best_node

    def score(self, X, y, /):
        """Evaluate the decision tree model on the test set.

        The decision tree regressor makes predictions for each test sample
        and outputs the mean squared error.

        Parameters
        ----------
        X : pd.DataFrame.dtypes
            The feature matrix of the dataset.
        y : pd.Series.dtypes
            The target vector of the dataset.

        Returns
        -------
        float
        """
        X, y = _to_pandas_dataframe(X, y)
        y_hat = [self.predict(X.iloc[i]) for i in range(len(X))]
        return mean_squared_error(y, y_hat, squared=False)


class ParallelDecisionTreeClassifier(DecisionTreeClassifier):
    """A parallel decision tree classifier.

    Refer to `DecisionTreeClassifier` for a more detailed explanation.

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.
    """

    def fit(self, X, y, /):
        """Fit the decision tree classifier to the dataset.

        The `fit` method induces a decision tree model through a greedy
        search in the hypothesis space. All processes are synchronized
        and maintains their execution times.

        Parameters
        ----------
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.

        Returns
        -------
        self
            The current instance of the `DecisionTreeClassifier`.
        """
        WORLD_COMM.Barrier()
        super().fit(X, y)

    def _make_tree(
        self, X, y, /, comm=WORLD_COMM, *, parent_y=None, branch=None, depth=0
    ):
        """Perform the ID3 (Iterative Dichotomiser 3) algorithm.

        See `DecisionTreeClassifier.make_tree` for a more detailed
        explanation.

        Parameters
        __________
        X, y : pd.DataFrame.dtypes
            The feature matrix and target vector of the dataset.

        comm : mpi4py.MPI.Intracomm, default=world_comm
            The current communicator of a processes.

        parent_y : pandas.core.series.Series, default=None
            The parent target feature.

        branch : str, default=None
            The transition from which the parent parition.

        depth : int, default=0
            The current level at a node.

        Returns
        -------
        best_node : Node
            The root node of the parallel decision tree classifier.
        """

        def make_node(value):
            return DecisionNode(feature=value, branch=branch, depth=depth)

        size = comm.Get_size()

        if len(np.unique(y)) == 1:
            return make_node(mode(y))
        if X.empty:
            return make_node(mode(parent_y))
        if all((X[d] == X[d].iloc[0]).all() for d in X.columns):
            return make_node(mode(y))
        if self.criterion.get("max_depth", np.inf) <= depth:
            return make_node(mode(y))
        if self.criterion.get("min_samples_split", -np.inf) >= len(X):
            return make_node(mode(y))

        with ThreadPool() as pool:
            max_gain = np.argmax(
                pool.map(lambda d: self.find_information_gain(X, y, d), X.columns)
            )

        if self.criterion.get("min_gain", -np.inf) >= max_gain:
            return make_node(mode(y))

        best_feature = X.columns[max_gain]
        best_node = deepcopy(make_node(best_feature))

        if is_numeric_dtype(X[best_feature]):
            best_node.threshold = round(self.n_thresholds[best_feature], 2)

        levels = [
            self.partition_data(X, y, best_feature, level, best_node.threshold)
            for level in self.n_levels[best_feature]
        ]

        if size == 1:
            for *d, level in levels:
                best_node.add(
                    self._make_tree(
                        *d, comm=comm, parent_y=y, branch=level, depth=depth + 1
                    )
                )
        else:
            r = comm.Get_rank()
            p = len(levels)

            group = MPI.Get_cyclic_dist(comm, n_block=p)

            *partition_data, level = levels[r % p]

            sub_tree = comm.allgather(
                {
                    level: self._make_tree(
                        *partition_data,
                        comm=group,
                        parent_y=y,
                        branch=level,
                        depth=depth + 1,
                    )
                }
            )

            for d in sub_tree:
                for level, node in d.items():
                    best_node.children[level] = node
                    node.parent = best_node

            group.Free()
        return best_node
