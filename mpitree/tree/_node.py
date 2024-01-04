""""""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(kw_only=True)
class Node:
    """A tree node class.

    Parameters
    ----------
    value: int
        The feature or class value.

    threshold : float, optional
        The value indicating the split boundaries.

    depth : int, default=0
        The number of levels from the root to a node.

    parent : Node, optional
        The precedent node (the default is `None`, which implies the node
        is a root node).

    left, right : Node, optional
        The node whose `value` is (less than or equal) or (greater) than
        its parent (the default is `None` which implies the node is a leaf
        node).
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

        Each line outputs a node indented according to their level with
        values either a feature (interior nodes) or class (leaf nodes),
        value and identified by the branch taken from its parent.

        Returns
        -------
        str
            The string-formatted node.
        """
        spacing = self.depth * "│  " + (
            "└──" if self.is_leaf else "├──" if self.depth else "┌──"
        )

        root_node_template = "{spacing} {value}"
        interior_node_template = "{spacing} {value} [{sign} {threshold}]"

        value_name = f"class: {self.value}" if self.is_leaf else f"feature_{self.value}"

        if not self.depth:
            return root_node_template.format(spacing=spacing, value=value_name)

        return interior_node_template.format(
            spacing=spacing,
            value=value_name,
            sign=("<=" if self is self.parent.left else ">"),
            threshold=round(self.parent.threshold, 2),
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
