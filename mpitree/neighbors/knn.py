import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, euclidean_distances, mean_squared_error
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """A K-nearest neighbors classifier implementation.

    Parameters
    ----------
    n_neighbors : int, default=5
        A hyperparameter specifing the (positive) number of closest training instances.
    """

    def __init__(self, *, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Store the training instances.

        Parameters
        ----------
        X : ndarray
            2D feature matrix with shape (n_samples, n_features) of
            numerical values.

        y : ndarray
            1D target array with shape (n_samples,) of categorical values.

        Raises
        ------
        ValueError
            If `n_neighbors` attribute is neither an integer nor between
            one and the feature matrix length inclusive.
        """
        X, y = check_X_y(X, y)

        if not (isinstance(self.n_neighbors, int) and 1 <= self.n_neighbors <= len(X)):
            raise ValueError(
                "`n_neighbors` attribute must be an integer between one and the feature matrix length inclusive."
            )

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Assign approximate categorical values to the test set.

        Parameters
        ----------
        X : ndarray
            2D test feature matrix with shape (n_samples, n_features) of
            numerical values.

        Returns
        -------
        np.ndarray
        """
        check_is_fitted(self)
        X = check_array(X)

        closest = np.argsort(euclidean_distances(X, self.X_), axis=1)[:, : self.n_neighbors]
        neighbors = mode(self.y_[closest], axis=1, keepdims=False)
        return neighbors.mode

    def score(self, X, y):
        """Evaluate the performance of a K-nearest neighbors classifier.

        The metric used to evaluate the performance of a K-nearest
        neighbors classifier is `sklearn.metrics.accuracy_score`.

        Parameters
        ----------
        X : ndarray
            2D test feature matrix with shape (n_samples, n_features) of
            numerical values.

        y : ndarray
            1D test target array with shape (n_samples,) of categorical
            values.
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y)

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class KNeighborsRegressor(BaseEstimator, RegressorMixin):
    """A K-nearest neighbors regressor implementation.

    Parameters
    ----------
    n_neighbors : int, default=5
        A hyperparameter specifing the (positive) number of closest training instances.
    """

    def __init__(self, *, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Store the training instances.

        Parameters
        ----------
        X : ndarray
            2D feature matrix with shape (n_samples, n_features) of
            numerical values.

        y : ndarray
            1D target array with shape (n_samples,) of numerical values.

        Raises
        ------
        ValueError
            If `n_neighbors` attribute is neither an integer nor between
            one and the feature matrix length inclusive.
        """
        X, y = check_X_y(X, y, y_numeric=True)

        if not (isinstance(self.n_neighbors, int) and 1 <= self.n_neighbors <= len(X)):
            raise ValueError(
                "`n_neighbors` attribute must be an integer between one and the feature matrix length inclusive."
            )

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Assign approximate numerical values to the test set.

        Parameters
        ----------
        X : ndarray
            2D test feature matrix with shape (n_samples, n_features) of
            numerical values.

        Returns
        -------
        np.ndarray
        """
        check_is_fitted(self)
        X = check_array(X)

        closest = np.argsort(euclidean_distances(X, self.X_), axis=1)[
            :, : self.n_neighbors
        ]
        return np.mean(self.y_[closest], axis=1)

    def score(self, X, y):
        """Evaluate the performance of a K-nearest neighbors regressor.

        The metric used to evaluate the performance of a K-nearest
        neighbors regressor is `sklearn.metrics.mean_square_error`.

        Parameters
        ----------
        X : ndarray
            2D test feature matrix with shape (n_samples, n_features) of
            numerical values.

        y : ndarray
            1D test target array with shape (n_samples,) of numerical
            values.
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y, y_numeric=True)

        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)

if __name__ == "__main__":
    from sklearn.datasets import load_diabetes, load_iris
    from sklearn.model_selection import train_test_split

    iris, diabetes = load_iris(), load_diabetes()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data[:, :2], iris.target, test_size=0.2, random_state=42
    )

    clf = KNeighborsClassifier().fit(X_train, y_train)
    print(clf.score(X_test, y_test).round(2))

    clf = KNeighborsClassifier(n_neighbors=1)
    clf = clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test).round(2))

    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=42
    )

    regr = KNeighborsRegressor().fit(X_train, y_train)
    print(regr.score(X_test, y_test).round(2))

    regr = KNeighborsRegressor(n_neighbors=1)
    regr = regr.fit(X_train, y_train)
    print(regr.score(X_test, y_test).round(2))