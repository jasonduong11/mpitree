#!/usr/bin/env python3
# pylint: skip-file

import sys
import unittest

sys.path.append("..")

from mpitree._node import Node


class TestNode(unittest.TestCase):
    def test_node_str(self):
        # TODO
        pass

    def test_node_eq(self):
        self.assertTrue(Node() == Node())
        self.assertFalse(Node(value="Alice") == Node(value="Bob"))
        with self.assertRaises(TypeError):
            Node() == "Not a Node"

    def test_node_add(self):
        self.assertFalse((Node() + Node(branch="< 0")).is_leaf)
        with self.assertRaises(AttributeError):
            Node(branch="a") + Node()
        with self.assertRaises(TypeError):
            Node() + "Not a Node"

    def test_node_is_leaf(self):
        self.assertTrue(Node().is_leaf)
        self.assertFalse(Node(children={"Alice": Node()}).is_leaf)
        with self.assertRaises(TypeError):
            Node(children="Not a Dict").is_leaf

    def test_node_left_property(self):
        self.assertIsNone(Node().left)
        self.assertEqual("Alice", Node(threshold=0, children={"< 0": "Alice"}).left)
        with self.assertRaises(TypeError):
            Node(children="Not a Dict").left

    def test_node_right_property(self):
        self.assertIsNone(Node().right)
        self.assertEqual("Bob", Node(threshold=0, children={">= 0": "Bob"}).right)
        with self.assertRaises(TypeError):
            Node(children="Not a Dict").left


if __name__ == "__main__":
    unittest.main(verbosity=2)
