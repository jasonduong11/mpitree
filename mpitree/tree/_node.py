""""""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

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
        Add Description Here.

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

    target: array-like
        1D dataset array with shape (n_samples,) of either categorical or
        numerical values.
    """

    feature: Union[str, float] = None
    threshold: Optional[float] = None
    branch: str = None
    depth: int = field(default_factory=int)
    parent: Optional[DecisionNode] = field(default=None, repr=False)
    children: dict = field(default_factory=dict, repr=False)
    target: np.ndarray = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.parent is not None:  # NOTE: root node default to 0
            self.depth = self.parent.depth + 1

    def __str__(self):
        """Export a string-formatted decision node.

        Each decision node is prefixed by one of three branch types
        specified for root, internal, and leaf decision nodes and is
        followed by their corresponding feature or target value. Each
        internal and leaf decision node displays a unique branch respective
        of the parent split node.

        Returns
        -------
        str
            The string-formatted decision node.
        """

        spacing = self.depth * "│  " + (
            "└──" if self.is_leaf else "├──" if self.depth else "┌──"
        )

        root_fmt = "{spacing} {feature}"
        interior_fmt = "{spacing} {feature} [{sign} {threshold}]"

        feature_name = str(self.feature) if self.is_leaf else f"feature_{self.feature}"

        if not self.depth:  # NOTE: the root node could be a leaf node.
            return root_fmt.format(spacing=spacing, feature=feature_name)

        return interior_fmt.format(
            spacing=spacing,
            feature=feature_name,
            sign=self.branch,
            threshold=round(self.parent.threshold, 2),
        )

    @property
    def is_leaf(self):
        """Return whether a node is terminal.

        A `DecisionNode` object is a leaf if it contains no children, and
        will return true; otherwise, the `DecisionNode` is considered an
        internal and will return false.

        Returns
        -------
        bool
        """
        return not self.children

    @property
    def left(self):
        return self.children["<="]

    @property
    def right(self):
        return self.children[">"]
