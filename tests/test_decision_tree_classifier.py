#!/usr/bin/env python3
# pylint: skip-file

import sys
import unittest

import pandas as pd

sys.path.append("..")

from mpitree.decision_tree import DecisionTreeClassifier

# ref: https://stackoverflow.com/questions/43842675/how-to-prevent-truncating-of-string-in-unit-test-python
if "unittest.util" in __import__("sys").modules:
    # Show full diff in self.assertEqual.
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999_999_999


def generate_data(features: dict) -> tuple:
    data = pd.DataFrame(features)
    return data.iloc[:, :-1], data.iloc[:, -1]


class TestDecisionTreeEstimator(unittest.TestCase):
    def test_eq(self):
        X, y = generate_data({"A": ["a"], "y": ["+"]})
        tree_a = DecisionTreeClassifier().fit(X, y)
        X, y = generate_data({"A": ["a"], "y": ["-"]})
        tree_b = DecisionTreeClassifier().fit(X, y)

        with self.assertRaises(TypeError):
            tree_a == "Not a Decision Tree"

        with self.assertRaises(AttributeError):
            DecisionTreeClassifier() == tree_a

        self.assertTrue(tree_a == tree_a)
        self.assertFalse(tree_a == tree_b)

    def test_str_with_tree_not_fitted(self):
        with self.assertRaises(AttributeError):
            str(DecisionTreeClassifier())

    def test_check_is_fitted(self):
        X, y = generate_data({"A": ["a"], "y": ["+"]})
        self.assertTrue(DecisionTreeClassifier().fit(X, y).check_is_fitted)
        self.assertFalse(DecisionTreeClassifier().check_is_fitted)

    def test_check_valid_params(self):
        with self.assertRaises(KeyError):
            DecisionTreeClassifier(criterion={"Invalid Key": 0}).fit(
                *generate_data({"A": [1], "y": [1]})
            )
            DecisionTreeClassifier(criterion={"Invalid Key": 0, "max_depth": 2}).fit(
                *generate_data({"A": [1], "y": [1]})
            )
        with self.assertRaises(Exception):
            DecisionTreeClassifier().fit(*generate_data({"A": [], "y": []}))
        with self.assertRaises(TypeError):
            tree_1 = DecisionTreeClassifier()
            tree_1.criterion = "str"

            tree_2 = DecisionTreeClassifier()
            tree_2._n_thresholds = "str"

            tree_1.fit(*generate_data({"A": [1], "y": [1]}))
            tree_2.fit(*generate_data({"A": [1], "y": [1]}))
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(criterion={"max_depth": -1}).fit(
                *generate_data({"A": [1], "y": [1]})
            )

    def test_max_depth(self):
        X, y = generate_data({"A": [1, 2, 3, 4], "B": [7, 8, 5, 6], "y": [1, 2, 1, 2]})
        tree = DecisionTreeClassifier(criterion={"max_depth": 1}).fit(X, y)
        self.assertTrue(max(node.depth for node in tree) == 1)


class TestDecisionTreeClassifier(unittest.TestCase):
    def test_find_entropy(self):
        X, y = generate_data({"A": ["a"], "y": ["+"]})
        self.assertEqual(DecisionTreeClassifier().find_entropy(X, y), 0)

        X, y = generate_data({"A": ["a", "a"], "y": ["+", "-"]})
        self.assertEqual(DecisionTreeClassifier().find_entropy(X, y), 1)

    def test_find_rem(self):
        X, y = generate_data({"A": ["a", "b"], "y": ["+", "-"]})
        self.assertEqual(DecisionTreeClassifier().find_rem(X, y, "A"), 0)

        X, y = generate_data({"A": ["a", "a"], "y": ["+", "-"]})
        self.assertEqual(DecisionTreeClassifier().find_rem(X, y, "A"), 1)

    def test_find_optimal_threshold(self):
        X, y = generate_data({"A": [0, 1, 2, 3], "y": ["0", "1", "0", "1"]})
        self.assertEqual(
            DecisionTreeClassifier().find_optimal_threshold(X, y, "A")[1],
            0.5,
        )

        X, y = generate_data({"A": [3, 0, 1, 0], "y": ["1", "0", "0", "0"]})
        self.assertEqual(
            DecisionTreeClassifier().find_optimal_threshold(X, y, "A")[1],
            2,
        )

    def test_find_information_gain(self):
        X, y = generate_data({"A": ["a", "a"], "y": ["+", "-"]})
        self.assertEqual(
            DecisionTreeClassifier().find_information_gain(X, y, "A"),
            0,
        )

        X, y = generate_data({"A": ["a", "b"], "y": ["+", "-"]})
        self.assertEqual(
            DecisionTreeClassifier().find_information_gain(X, y, "A"),
            1,
        )

        tree = DecisionTreeClassifier()
        # NOTE: must initialize because we not call the `fit` method
        tree._n_thresholds = {}

        X, y = generate_data({"A": [0, 0, 1, 1], "y": ["0", "0", "1", "1"]})
        self.assertEqual(tree.find_information_gain(X, y, "A"), 1)

        X, y = generate_data({"A": [1, 0, 1, 0], "y": ["0", "1", "0", "1"]})
        self.assertEqual(tree.find_information_gain(X, y, "A"), 1)

    def test_fit_make_tree_with_categorical_variables(self):
        # CR-root is leaf with identical target values (Note: trivial with one datapoint)
        X, y = generate_data({"A": ["a", "a"], "y": ["+", "+"]})
        self.assertEqual(str(DecisionTreeClassifier().fit(X, y)), "└── +")

        # CR-root is leaf with identical feature values
        X, y = generate_data({"A": ["a", "a", "a"], "y": ["+", "+", "-"]})
        self.assertEqual(str(DecisionTreeClassifier().fit(X, y)), "└── +")

        # C-split with irrelevant feature
        X, y = generate_data({"A": ["a", "b"], "B": ["a", "a"], "y": ["+", "-"]})
        self.assertEqual(
            str(DecisionTreeClassifier().fit(X, y)),
            "┌── A\n│  └── + [a]\n│  └── - [b]",
        )

        # CC-top-split
        X, y = generate_data(
            {
                "A": ["a", "a", "a", "b"],
                "B": ["c", "c", "d", "d"],
                "y": ["+", "-", "+", "-"],
            }
        )
        self.assertEqual(
            str(DecisionTreeClassifier().fit(X, y)),
            "┌── A\n│  ├── B [a]\n│  │  └── + [c]\n│  │  └── + [d]\n│  └── - [b]",
        )

    def test_fit_make_tree_with_continuous_variables(self):
        # NR-root is leaf with identical target values
        X, y = generate_data({"A": [-1, 1], "y": ["+", "+"]})
        self.assertEqual(str(DecisionTreeClassifier().fit(X, y)), "└── +")

        # NR-root is leaf with identical feature values
        X, y = generate_data({"A": [0, 0], "y": ["+", "-"]})
        self.assertEqual(str(DecisionTreeClassifier().fit(X, y)), "└── +")

        # N-split
        X, y = generate_data({"A": [0, 1], "B": [0, 0], "y": ["+", "-"]})
        self.assertEqual(
            str(DecisionTreeClassifier().fit(X, y)),
            "┌── A\n│  └── + [< 0.5]\n│  └── - [>= 0.5]",
        )

        # NN-top-split
        X, y = generate_data(
            {"A": [0, 0, 0, 1], "B": [0, 0, 1, 1], "y": ["+", "-", "+", "-"]}
        )
        self.assertEqual(
            str(DecisionTreeClassifier().fit(X, y)),
            "┌── A\n│  ├── B [< 0.5]\n│  │  └── + [< 0.5]\n│  │  └── + [>= 0.5]\n│  └── - [>= 0.5]",
        )

    def test_fit_make_tree_with_categorical_and_continuous_variables(self):
        # CN-bot-split
        X, y = generate_data(
            {"A": ["a", "b", "b", "b"], "B": [0, 0, 1, 1], "y": ["+", "-", "+", "-"]}
        )
        self.assertEqual(
            str(DecisionTreeClassifier().fit(X, y)),
            "┌── A\n│  └── + [a]\n│  ├── B [b]\n│  │  └── - [< 0.5]\n│  │  └── + [>= 0.5]",
        )

        # NC-bot-split
        X, y = generate_data(
            {"A": [0, 1, 1, 1], "B": ["a", "a", "b", "b"], "y": ["+", "-", "+", "-"]}
        )
        self.assertEqual(
            str(DecisionTreeClassifier().fit(X, y)),
            "┌── A\n│  └── + [< 0.5]\n│  ├── B [>= 0.5]\n│  │  └── - [a]\n│  │  └── + [b]",
        )
