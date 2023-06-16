"""This module contains the node class."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Union

import numpy as np


@dataclass(kw_only=True)
class DecisionNode:
    """A decision tree node.

    The decision tree node defines the attributes and properties of a
    `BaseDecisionTree`.

    Parameters
    ----------
    feature : str or float, default=None
        The descriptive or target feature value.

    threshold : float, default=None
        The default is `None`, which implies the split feature is
        categorical).

    branch : str, default=None
        The feature value of a split from the parent node.

    depth : int, default=0
        The number of levels from the root to a node. The root node `depth`
        is initialized to 0 and successor nodes are one depth lower from
        its parent.

    parent : DecisionNode, optional
        The precedent node.

    children : OrderedDict, default={}
        The nodes on each split of the parent node.

    state : np.ndarray
        2D dataset array with shape (n_samples, n_features+1) of
        categorical and/or numerical values.

    value : np.ndarray, init=False
        1D array with shape (n_classes,) of categorical (classification)
        values containing the number of instances for each classes in
        `state`.

    n_samples : int, init=False
        The number of instances in the dataset `state`.

    classes : np.ndarray
        Add Description Here.

    Notes
    -----
    .. note::
        The `threshold` attribute is initialized upon the split of a
        numerical feature.

        The `branch` attribute is assigned a value from the set of unique
        feature values for *categorical* features and `["True" | "False"]`
        for *numerical* features.
    """

    _estimator_type = ClassVar[str]

    feature: Union[str, float] = None
    threshold: Optional[float] = None
    branch: str = None
    depth: int = field(default_factory=int)
    parent: Optional[DecisionNode] = field(default=None, repr=False)
    children: OrderedDict = field(default_factory=dict, repr=False)
    state: np.ndarray = field(default_factory=list, repr=False)
    value: np.ndarray = field(init=False)
    n_samples: int = field(init=False)

    # NOTE: contingent to future changes
    classes: np.ndarray = field(default_factory=list, repr=False)
    feature_indices: np.ndarray = field(default_factory=list)

    def __post_init__(self):
        n_unique_class = dict(zip(*np.unique(self.y, return_counts=True)))

        self.value = np.array([n_unique_class.get(c, 0) for c in self.classes])
        self.n_samples = len(self.y)

        if self.parent is not None:
            self.depth = self.parent.depth + 1

    def __str__(self):
        """Export a string-formatted decision node.

        The output string of a node is primarily dependent on the `depth`
        for horizontal spacing and `branch`, either an interior or leaf.

        Returns
        -------
        str
            The string-formatted decision node.
        """

        spacing = self.depth * "│  " + (
            "└── " if self.is_leaf else "├── " if self.depth else "┌── "
        )

        feature = self.feature
        branch = self.branch

        if not self.depth:
            if self.is_leaf and self._estimator_type == "classifier":
                return spacing + f"class: {feature}"
            if self.is_leaf and self._estimator_type == "regressor":
                return spacing + f"target: {feature}"
            return spacing + str(feature)

        if self.parent and self.parent.threshold:
            # NOTE: numpy cannot have mix types so numerical value are
            # type casted to ``class <str>``
            if self.branch == "True":
                branch = f"<= {float(self.parent.threshold):.2f}"
            else:
                branch = f"> {float(self.parent.threshold):.2f}"

        if self.is_leaf and self._estimator_type == "classifier":
            return spacing + f"class: {feature} [{branch}]"
        if self.is_leaf and self._estimator_type == "regressor":
            return spacing + f"target: {feature} [{branch}]"

        return spacing + f"{feature} [{branch}]"

    def __lt__(self, other: DecisionNode = None):
        """Short Summary

        Extended Summary

        Parameters
        ----------
        other : DecisionNode, default=None

        Returns
        -------
        bool

        Notes
        -----
        The `other` parameter is unsed.
        """
        return self.is_leaf

    @property
    def y(self):
        """Short Summary

        Extended Summary

        Returns
        -------
        np.ndarray
        """
        return self.state[:, -1]

    @property
    def proba(self):
        """Short Summary

        Extended Summary

        Returns
        -------
        np.ndarray

        Notes
        -----
        if dataset is empty (``n_samples=0``), the probability would be
        zero for each class as default.
        """
        if self.n_samples == 0:
            return np.zeros(len(self.classes))
        return self.value / self.n_samples

    @property
    def is_leaf(self):
        """Return whether a node is terminal.

        A `DecisionNode` object is a leaf if it contains no children, and
        will return true; otherwise, the `DecisionNode` is considered an
        internal and will return false.

        Returns
        -------
        bool

        Raises
        ------
        TypeError
            If `children` attribute is not type `dict`
        """
        if not isinstance(self.children, dict):
            raise TypeError(f"Expected `dict` type but got: {type(self.children)}")
        return not self.children

    @property
    def left(self) -> DecisionNode:
        """Return the left child of a numeric-featured decision node.

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
        return self.children.get("True")

    @property
    def right(self) -> DecisionNode:
        """Return the right child of a numeric-featured decision node.

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
        return self.children.get("False")

    def add(self, other: DecisionNode):
        """Add another node to a existing node children.

        The operation will append another `DecisionNode` with its key, specified
        by its `branch` value, to an existing `DecisionNode` children dictionary.

        Parameters
        ----------
        other : DecisionNode
            The child decision node.

        Returns
        -------
        DecisionNode

        Raises
        ------
        TypeError
            If `other` is not type `DecisionNode`.
        AttributeError
            If `other` branch attribute is not instantiated.
        """
        if not isinstance(other, self.__class__):
            raise TypeError("Expected `DecisionNode` type but got: %s", type(other))
        if other.branch is None:
            raise AttributeError("Object's `branch` attribute is not instantiated")

        self.children[other.branch] = other
        return self
