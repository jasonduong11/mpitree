#!/usr/bin/env python3
# pylint: skip-file

import sys
import unittest

import pandas as pd

sys.path.append("..")

from mpitree.decision_tree import DecisionTreeRegressor

# ref: https://stackoverflow.com/questions/43842675/how-to-prevent-truncating-of-string-in-unit-test-python
if "unittest.util" in __import__("sys").modules:
    # Show full diff in self.assertEqual.
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999_999_999


def generate_data(features: dict) -> tuple:
    data = pd.DataFrame(features)
    return data.iloc[:, :-1], data.iloc[:, -1]


class TestDecisionTreeRegressor(unittest.TestCase):
    def test_find_variance(self):
        X, y = generate_data({"A": ["0"], "y": [0]})
        self.assertEqual(DecisionTreeRegressor().find_variance(X, y), 0)

        X, y = generate_data({"A": ["0", "0"], "y": [0, 1]})
        self.assertEqual(DecisionTreeRegressor().find_variance(X, y), 0.5)

    def test_find_weighted_variance(self):
        X, y = generate_data({"A": ["0", "0", "1", "1"], "y": [0, 1, 0, 1]})
        self.assertEqual(DecisionTreeRegressor().find_weighted_variance(X, y, "A"), 0.5)

    def test_fit_make_tree_with_categorical_variables(self):
        # CR-root is leaf with identical target values (Note: trivial with one datapoint)
        X, y = generate_data({"A": ["a", "a"], "y": [0, 0]})
        self.assertEqual(str(DecisionTreeRegressor().fit(X, y)), "└── 0")

        # CR-root is leaf with identical feature values
        X, y = generate_data({"A": ["a", "a", "a"], "y": [0, 0, 1]})
        self.assertEqual(str(DecisionTreeRegressor().fit(X, y)), "└── 0")

        # C-split with irrelevant feature
        X, y = generate_data({"A": ["a", "b"], "B": ["a", "a"], "y": [0, 1]})
        self.assertEqual(
            str(DecisionTreeRegressor().fit(X, y)),
            "┌── A\n│  └── 0 [a]\n│  └── 1 [b]",
        )

        # CC-top-split
        X, y = generate_data(
            {"A": ["a", "a", "a", "b"], "B": ["c", "c", "d", "d"], "y": [0, 1, 0, 1]}
        )
        self.assertEqual(
            str(DecisionTreeRegressor().fit(X, y)),
            "┌── A\n│  ├── B [a]\n│  │  └── 0.5 [c]\n│  │  └── 0 [d]\n│  └── 1 [b]",
        )

    def test_fit_make_tree_with_continuous_variables(self):
        # NR-root is leaf with identical target values
        X, y = generate_data({"A": [0], "y": [1]})
        self.assertEqual(str(DecisionTreeRegressor().fit(X, y)), "└── 1")

        # NR-root is leaf with identical feature values
        X, y = generate_data({"A": [0, 0], "y": [0, 1]})
        self.assertEqual(str(DecisionTreeRegressor().fit(X, y)), "└── 0")

        # N-split
        X, y = generate_data({"A": [0, 1], "B": [0, 0], "y": [0, 1]})
        self.assertEqual(
            str(DecisionTreeRegressor().fit(X, y)),
            "┌── A\n│  └── 0 [< 0.5]\n│  └── 1 [>= 0.5]",
        )

        # NN-top-split
        X, y = generate_data({"A": [0, 0, 0, 1], "B": [2, 2, 3, 3], "y": [0, 1, 0, 1]})
        self.assertEqual(
            str(DecisionTreeRegressor().fit(X, y)),
            "┌── A\n│  ├── B [< 0.5]\n│  │  └── 0 [< 2.5]\n│  │  └── 0 [>= 2.5]\n│  └── 1 [>= 0.5]",
        )

    def test_fit_make_tree_with_categorical_and_continuous_variables(self):
        # CN-bot-split
        X, y = generate_data(
            {"A": ["a", "b", "b", "b"], "B": [0, 0, 1, 1], "y": [0, 1, 0, 1]}
        )
        self.assertEqual(
            str(DecisionTreeRegressor().fit(X, y)),
            "┌── A\n│  └── 0 [a]\n│  ├── B [b]\n│  │  └── 1 [< 0.5]\n│  │  └── 0 [>= 0.5]",
        )

        # NC-bot-split
        X, y = generate_data(
            {"A": [0, 0, 0, 1], "B": ["a", "a", "b", "b"], "y": [0, 1, 0, 1]}
        )
        self.assertEqual(
            str(DecisionTreeRegressor().fit(X, y)),
            "┌── B\n│  └── 0 [a]\n│  ├── A [b]\n│  │  └── 0 [< 0.5]\n│  │  └── 1 [>= 0.5]",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
