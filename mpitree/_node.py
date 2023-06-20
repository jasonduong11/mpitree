"""This module contains the node class."""

from __future__ import annotations

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

    children : dict, default={}
        The nodes on each split of the parent node.

    state : np.ndarray
        2D dataset array with shape (n_samples, n_features + 1) of either
        or both categorical and numerical values.

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
    children: dict = field(default_factory=dict, repr=False)
    state: np.ndarray = field(default_factory=list, repr=False)
    value: np.ndarray = field(init=False)
    n_samples: int = field(init=False)

    # NOTE: contingent to future changes
    classes: np.ndarray = field(default_factory=list, repr=False)
    # NOTE: only for debugging, remove later
    feature_indices: np.ndarray = field(default_factory=list)

    def __post_init__(self):
        n_unique_class = dict(zip(*np.unique(self.y, return_counts=True)))
        self.value = np.array([n_unique_class.get(k, 0) for k in self.classes])

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
            "└──" if self.is_leaf else "├──" if self.depth else "┌──"
        )

        info = self.feature

        if self.is_leaf and self._estimator_type == "classifier":
            info = f"class: {self.feature}"
        if self.is_leaf and self._estimator_type == "regressor":
            info = f"target: {self.feature}"

        if not self.parent:
            # for root node, we don't include `branch`
            # NOTE: the root node could be a leaf node too
            return f"{spacing} {info}"
        elif self.parent.threshold:
            # NOTE: numpy cannot have mix types so numerical value are
            # type casted to ``class <str>``
            if self.branch == "True":
                branch = f"<= {float(self.parent.threshold):.2f}"
            else:
                branch = f"> {float(self.parent.threshold):.2f}"
        else:
            branch = self.branch  # for categorical features

        return f"{spacing} {info} [{branch}]"

    def __getitem__(self, branch: Union[str, float]) -> DecisionNode:
        if self.threshold is not None:
            branch = ("True", "False")[branch <= self.threshold]
        return self.children[branch]

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
        """
        self.children[other.branch] = other
        return self
