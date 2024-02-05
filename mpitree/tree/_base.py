"""This module defines the node class.

Author: Jason Duong
Date: 01/11/24
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from numpy.typing import ArrayLike

from enum import Enum


class BranchType(Enum):
    ROOT = "┌──"
    INTERIOR_LIKE = "├──"
    LEAF_LIKE = "└──"


@dataclass(kw_only=True)
class Node:
    """A tree node class.

    Parameters
    ----------
    value: int
        The feature index or target value.

    threshold : float, optional
        The feature value representing the split boundary.

    depth : int, default=0
        The number of levels from the root to a node.

    count : ArrayLike
        The array of occurrences in for each class label.

    sign : str, optional
        The region indicator of the split boundary.

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
    count: ArrayLike = field(default=None)
    sign: Optional[str] = field(default=None, repr=False)
    parent: Optional[Node] = field(default=None, repr=False)
    left: Optional[Node] = field(default=None, repr=False)
    right: Optional[Node] = field(default=None, repr=False)

    _btype: BranchType = field(default=BranchType.ROOT, repr=False)

    def __post_init__(self):
        if self.parent is not None:
            self.depth = self.parent.depth + 1

    def __lt__(self, other: Node):
        # when both self (>) and other (<=) are leaf nodes or just other is an interior node
        if self.is_leaf:
            other._btype = BranchType.INTERIOR_LIKE
            self._btype = BranchType.LEAF_LIKE

        else:  # when both self (>) and other (<=) are interior nodes or just self (>) is an interior node
            self._btype = BranchType.INTERIOR_LIKE
            other._btype = BranchType.LEAF_LIKE

        return not self.is_leaf

    def __str__(self):
        """Return a string-formatted node.

        The function outputs a node's `value` and the branch taken from its
        parent.

        Returns
        -------
        str
        """

        value = f"class: {self.value}" if self.is_leaf else f"feature_{self.value}"

        if not self.depth:
            return value

        # NOTE: this is only because mpi.allgather returns a copy, not reference
        if self.sign is None:
            sign = "<=" if self is self.parent.left else ">"
        else:
            sign = self.sign

        return f"{value} [{sign} {self.parent.threshold:.2f}]"

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
        return [self.left, self.right] if not self.is_leaf else []
