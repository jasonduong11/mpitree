#!/usr/bin/env python3
# pylint: skip-file

import sys
import unittest

sys.path.append("..")

from mpitree._node import DecisionNode


class TestNode(unittest.TestCase):
    def test_node_str(self):
        # TODO
        pass

    def test_node_eq(self):
        self.assertTrue(DecisionNode() == DecisionNode())
        self.assertFalse(DecisionNode(feature="Alice") == DecisionNode(feature="Bob"))
        with self.assertRaises(TypeError):
            DecisionNode() == "Not a Node"

    def test_node_add(self):
        self.assertFalse((DecisionNode() + DecisionNode(branch="< 0")).is_leaf)
        with self.assertRaises(AttributeError):
            DecisionNode(branch="a") + DecisionNode()
        with self.assertRaises(TypeError):
            DecisionNode() + "Not a Node"

    def test_node_is_leaf(self):
        self.assertTrue(DecisionNode().is_leaf)
        self.assertFalse(DecisionNode(children={"Alice": DecisionNode()}).is_leaf)
        with self.assertRaises(TypeError):
            DecisionNode(children="Not a Dict").is_leaf

    def test_node_left_property(self):
        self.assertIsNone(DecisionNode().left)
        self.assertEqual(
            "Alice", DecisionNode(threshold=0, children={"< 0": "Alice"}).left
        )
        with self.assertRaises(TypeError):
            DecisionNode(children="Not a Dict").left

    def test_node_right_property(self):
        self.assertIsNone(DecisionNode().right)
        self.assertEqual(
            "Bob", DecisionNode(threshold=0, children={">= 0": "Bob"}).right
        )
        with self.assertRaises(TypeError):
            DecisionNode(children="Not a Dict").left


if __name__ == "__main__":
    unittest.main(verbosity=2)
