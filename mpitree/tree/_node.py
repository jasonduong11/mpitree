""""""

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

    target: np.ndarray
        1D dataset array with shape (n_samples,) of either categorical or
        numerical values.

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
    target: np.ndarray = field(default_factory=list, repr=False)
    value: np.ndarray = field(init=False)
    n_samples: int = field(init=False)

    # NOTE: contingent to future changes
    classes: np.ndarray = field(default_factory=list, repr=False)

    def __post_init__(self):
        # TODO: refactor -> {0: 2, 2: 1} -> [2, 0, 1]
        n_class_dist = dict(zip(*np.unique(self.target, return_counts=True)))
        self.value = np.array([n_class_dist.get(k, 0) for k in self.classes])

        self.n_samples = len(self.target)

        if self.parent is not None:
            self.depth = self.parent.depth + 1

    def __str__(self):
        """Export a string-formatted decision node.

        Each decision node is prefixed by one of three branch types
        specified for root, internal, and leaf decision nodes and is
        followed by their corresponding feature or target value. Each
        internal and leaf decision node displays a unique branch respective
        of the parent split node. For leaf decision nodes, the target value
        is additionally prefixed by either "class" or "target" depending on
        the task (i.e., classification or regression).

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
            # NOTE: the root node could be a leaf node.
            return f"{spacing} {info}"

        if self.parent.threshold:
            # NOTE: Numpy cannot have mix types so numerical value are
            # type-casted to ``class <str>``.
            if self.branch == "True":
                branch = f"<= {float(self.parent.threshold):.2f}"
            else:
                branch = f"> {float(self.parent.threshold):.2f}"
        else:
            branch = self.branch  # for categorical features

        return f"{spacing} {info} [{branch}]"

    def __getitem__(self, other: str | float) -> DecisionNode:
        """Short Summary

        Extended Summary

        Parameters
        ----------
        other : str or float

        Returns
        -------
        DecisionNode
        """
        if self.threshold is not None:
            branch = ("True", "False")[other <= self.threshold]
        else:
            branch = other
        return self.children[branch]

    @property
    def is_leaf(self):
        """Return whether a node is terminal.

        A `DecisionNode` object is a leaf if it contains no children, and
        will return true; otherwise, the `DecisionNode` is considered an
        internal and will return false.

        Returns
        -------
        bool

        Notes
        -----
        The function is well-defined for all `DecisionNode` along a
        fully constructed decision path as the algorithm backtracks
        on a leaf decision node and each internal decision node has
        at least one child.
        """
        return not self.children