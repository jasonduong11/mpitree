"""This module contains the node class."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging
logger.basicConfig(
    level=logger.DEBUG, format="%(message)s", filename="debug.log", filemode="w"
)
logger.getLogger("matplotlib.font_manager").disabled = True


@dataclass(kw_only=True)
class DecisionNode:
    """A decision tree node.

    The decision tree node defines the attributes and properties of a
    ``DecisionTreeEstimator``.

    Parameters
    ----------
    feature : str or float, default=None
        The descriptive or target feature value.

    threshold : float, optional
        default is `None`, implying the split feature is *categorical*).

    branch : str, default=None
        The feature value of a split from the parent node.

    parent : Node, optional
        The precedent node.

    depth : int, default=0
        The number of levels from the root to a node.

    children : dict, default={}
        The nodes on each split of the parent node.

    shape : dict, default={}

    samples : int, default=0

    Notes
    -----
    .. note::
        The `threshold` attribute is initialized upon the split of a
        numerical feature.

        The `branch` attribute is assigned a value from the set of unique feature values
        for *categorical* features and is assigned ``[< | >=] threshold`` for
        *numerical* features.
    """

    feature: Union[str, float] = None
    threshold: Optional[float] = None
    branch: str = None
    parent: Optional[DecisionNode] = None
    depth: int = field(default_factory=int)
    children: dict = field(default_factory=dict)
    shape: dict = field(default_factory=dict)

    def __post_init__(self):
        self.samples = sum(self.shape.values())

    def __str__(self):
        """Output a string-formatted node.

        The output string of a node is primarily dependent on the `depth`
        for horizontal spacing and `branch`, either an interior or leaf.

        Returns
        -------
        str
            The string-formatted node.

        Raises
        ------
        ValueError
            If the `depth` is a negative integer.
        """
        if self.depth < 0:
            raise ValueError("Node depth attribute must be positive.")

        spacing = self.depth * "│  " + (
            "└── " if self.is_leaf else "├── " if self.depth else "┌── "
        )

        value = self.feature
        branch = self.branch
        if self.parent and self.parent.threshold:
            # FIXME: better solution to extract operator
            # possibly make a indirect branch property
            op, _ = self.branch.split(" ")
            branch = f"{op} {self.parent.threshold:.2f}"

        # def isfloat(value):
        #     try:
        #         float(value)
        #     except ValueError:
        #         return False
        #     return True

        if not self.depth:
            return spacing + str(value)

        # if self.is_leaf and isfloat(value):
        #     value = f"{float(value):.2}"

        return spacing + f"{value} [{branch}]"

    def __eq__(self, other: DecisionNode):
        """Check if two node objects are equivalent.

        Performs a pair-wise comparison among attributes of both `Node`
        objects and returns true if attributes are equal and returns false
        otherwise. The function will raise a TypeError if an object is not
        a `Node` instance.

        Parameters
        ----------
        other : Node
            The comparision object.

        Returns
        -------
        bool
            Returns true if both `Node` objects contain identical values
            for all attributes.

        Raises
        ------
        TypeError
            If `other` is not type `Node`.

        See Also
        --------
        DecisionTreeEstimator.__eq__ :
            Check if two decision tree objects are identical.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Expected `Node` type but got: {type(other)}")
        return self.__dict__ == other.__dict__

    def add(self, other: DecisionNode):
        """Add another node to a existing node children.

        The operation will append another `Node` with its key, specified
        by its `branch` value, to an existing `Node` children dictionary.

        Parameters
        ----------
        other : Node
            The comparision object.

        Returns
        -------
        self
            The current instance of the `Node` class.

        Raises
        ------
        TypeError
            If `other` is not type `Node`.
        Attribute Error
            If `other` branch attribute is not instantiated.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Expected `Node` type but got: {type(other)}")
        if other.branch is None:
            raise AttributeError("Object's `branch` attribute is not instantiated")

        other.parent = self
        self.children[other.branch] = other
        return self

    @property
    def is_leaf(self):
        """Return whether a node is terminal.

        A `Node` object is a leaf if it contains no children, and will
        return true; otherwise, the `Node` is considered an interior or
        root and will return false.

        Returns
        -------
        bool
            Returns true if the node is terminal.

        Raises
        ------
        TypeError
            If `children` attribute is not type `dict`

        Examples
        --------
        >>> from mpitree.base_estimator import Node
        >>> Node().is_leaf
        True
        >>> Node(children={"a": ...}).is_leaf
        False
        """
        if not isinstance(self.children, dict):
            raise TypeError(f"Expected `dict` type but got: {type(self.children)}")
        return not self.children

    @property
    def left(self):
        """Return the left child of a numeric-featured node.

        The `left` property accesses the first item (i.e., child)
        corresponding to the partition whose values for some feature
        is less than the specified `threshold`.

        Returns
        -------
        Node or None
            Returns a `Node` if its key exists in its parent first
            child; otherwise, returns None.

        Raises
        ------
        TypeError
            If `children` attribute is not type `dict`.
        """
        if not isinstance(self.children, dict):
            raise TypeError(f"Expected `dict` type but got: {type(self.children)}")
        return self.children.get(f"< {self.threshold}")

    @property
    def right(self):
        """Return the right child of a numeric-featured node.

        The `right` property accesses the second item (i.e., child)
        corresponding to the partition whose values for some feature
        is greater than or equal to the specified `threshold`.

        Returns
        -------
        Node or None
            Returns a `Node` if its key exists in its parent second
            child; otherwise, returns None.

        Raises
        ------
        TypeError
            If `children` attribute is not type `dict`.
        """
        if not isinstance(self.children, dict):
            raise TypeError(f"Expected `dict` type but got: {type(self.children)}")
        return self.children.get(f">= {self.threshold}")
