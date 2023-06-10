"""This module contains the node class."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Union, ClassVar

import numpy as np


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

    parent : DecisionNode, optional
        The precedent node.

    depth : int, default=0
        The number of levels from the root to a node.

        *** add information on post init rules ***

    children : OrderedDict, default={}
        The nodes on each split of the parent node.

        *** add information on ordering of branches ***

    shape : list, default=[]

    state : np.ndarray

    n_samples : int, default=0

    Notes
    -----
    .. note::
        The `threshold` attribute is initialized upon the split of a
        numerical feature.

        The `branch` attribute is assigned a value from the set of unique feature values
        for *categorical* features and is assigned ``[< | >=] threshold`` for
        *numerical* features.
    """

    _dtype = ClassVar[str]

    feature: Union[str, float] = None
    threshold: Optional[float] = None
    branch: str = None
    parent: Optional[DecisionNode] = None
    depth: int = field(default_factory=int)
    children: OrderedDict = field(default_factory=dict)
    shape: list = field(default_factory=list)
    state: np.ndarray

    def __post_init__(self):
        _, self.shape = np.unique(self.y, return_counts=True)
        self.n_samples = np.sum(self.shape)

        if self.parent is not None:
            self.depth = self.parent.depth + 1

    def __lt__(self, other: DecisionNode):
        """
        Notes
        -----
        `other` is not used
        """
        return self.is_leaf

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
            raise ValueError("DecisionNode `depth` attribute must be positive.")

        spacing = self.depth * "│  " + (
            "└── " if self.is_leaf else "├── " if self.depth else "┌── "
        )

        feature = self.feature
        branch = self.branch

        if self.parent and self.parent.threshold:
            # FIXME: better solution to extract operator
            # possibly make a indirect branch property
            op, _ = self.branch.split(" ")
            branch = f"{op} {self.parent.threshold:.2f}"

        if not self.depth:
            return spacing + str(feature)

        if self.is_leaf and self._dtype == "classifier":
            return spacing + f"class: {feature} [{branch}]"
        elif self.is_leaf and self._dtype == "regressor":
            return spacing + f"class: {feature} [{branch}]"

        return spacing + f"{feature} [{branch}]"

    @property
    def y(self):
        return self.state[:, -1]

    def add(self, other: DecisionNode):
        """Add another node to a existing node children.

        The operation will append another `DecisionNode` with its key, specified
        by its `branch` value, to an existing `DecisionNode` children dictionary.

        Parameters
        ----------
        other : DecisionNode
            The comparision object.

        Returns
        -------
        self
            The current instance of the `DecisionNode` class.

        Raises
        ------
        TypeError
            If `other` is not type `DecisionNode`.
        Attribute Error
            If `other` branch attribute is not instantiated.
        """
        if not isinstance(other, self.__class__):
            raise TypeError("Expected `DecisionNode` type but got: %s", type(other))
        if other.branch is None:
            raise AttributeError("Object's `branch` attribute is not instantiated")

        # other.parent = self
        self.children[other.branch] = other

        return self

    @property
    def is_leaf(self):
        """Return whether a node is terminal.

        A `DecisionNode` object is a leaf if it contains no children, and will
        return true; otherwise, the `DecisionNode` is considered an interior or
        root and will return false.

        Returns
        -------
        bool
            Returns true if the node is terminal.

        Raises
        ------
        TypeError
            If `children` attribute is not type `dict`
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
        DecisionNode or None
            Returns a `DecisionNode` if its key exists in its parent first
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
        DecisionNode or None
            Returns a `DecisionNode` if its key exists in its parent second
            child; otherwise, returns None.

        Raises
        ------
        TypeError
            If `children` attribute is not type `dict`.
        """
        if not isinstance(self.children, dict):
            raise TypeError(f"Expected `dict` type but got: {type(self.children)}")
        return self.children.get(f">= {self.threshold}")
