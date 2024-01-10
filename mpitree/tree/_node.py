"""
This module defines the tree node class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(kw_only=True)
class Node:
    """A tree node class.

    Parameters
    ----------
    value: int
        The feature index or target value.

    threshold : float, optional
        The feature value representing a split boundary.

    depth : int, default=0
        The number of levels from the root to a node.

    parent : Node, optional
        The precedent node (the default is `None`, which implies the node
        is a root node).

    left, right : Node, optional
        The node whose `value` is (less than or equal to) or (greater than)
        its `parent` (the default is `None` which implies the node is a
        leaf node).
    """

    value: int
    threshold: Optional[float] = None
    depth: int = field(default_factory=int)
    parent: Optional[Node] = field(default=None, repr=False)
    left: Optional[Node] = field(default=None, repr=False)
    right: Optional[Node] = field(default=None, repr=False)

    def __post_init__(self):
        if self.parent is not None:
            self.depth = self.parent.depth + 1

    def __str__(self):
        """Return a string-formatted node.

        The function outputs a node's `value` indented according to their
        `depth` and their split condition for non-root nodes.

        Returns
        -------
        str
        """
        spacing = self.depth * "│  " + (
            "└──" if self.is_leaf else "├──" if self.depth else "┌──"
        )

        root_node_signature = "{spacing} {value}"
        interior_node_signature = "{spacing} {value} [{sign} {threshold}]"

        value = f"class: {self.value}" if self.is_leaf else f"feature_{self.value}"

        if not self.depth:
            return root_node_signature.format(spacing=spacing, value=value)

        sign = "<=" if self is self.parent.left else ">"
        threshold = round(self.parent.threshold, 2)

        return interior_node_signature.format(
            spacing=spacing,
            value=value,
            sign=sign,
            threshold=threshold,
        )

    @property
    def is_leaf(self):
        """Return whether a node has no children nodes.

        The function returns true when both `left` and `right` attributes
        are `None`.

        Returns
        -------
        bool
        """
        return self.left is None and self.right is None

    def get_children(self):
        """Return a list of left and right nodes.

        If the node is a leaf node, the function returns an empty list;
        otherwise, returns a list of left and right nodes.

        Returns
        -------
        list
        """

        if self.is_leaf:
            return []
        return [self.left, self.right]
