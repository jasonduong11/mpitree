""""""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from numpy.typing import ArrayLike


@dataclass(kw_only=True)
class DecisionNode:
    """A decision tree node.

    The decision tree node defines the attributes and properties of a
    `BaseDecisionTree`.

    Parameters
    ----------
    value: str or float
        The descriptive or target feature value.

    threshold : float, optional
        Add Description Here.

    level : str, optional
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

    value: Union[str, float]
    threshold: Optional[float] = None
    level: Optional[str] = None
    depth: int = field(default_factory=int)
    parent: Optional[DecisionNode] = field(default=None, repr=False)
    children: dict = field(default_factory=dict, repr=False)
    target: Optional[ArrayLike] = field(default_factory=None, repr=False)

    def __post_init__(self):
        if self.parent is not None:
            self.depth = self.parent.depth + 1

    def __str__(self):
        """Export a string-formatted decision node.

        Returns
        -------
        str
            The string-formatted decision node.
        """

        spacing = self.depth * "│  " + (
            "└──" if self.is_leaf else "├──" if self.depth else "┌──"
        )

        root_tmp = "{spacing} {value}"
        interior_tmp = "{spacing} {value} [{sign} {threshold}]"

        feature_name = (
            f"class: {self.value}" if self.is_leaf else f"feature_{self.value}"
        )

        if not self.depth:
            return root_tmp.format(spacing=spacing, value=feature_name)

        return interior_tmp.format(
            spacing=spacing,
            value=feature_name,
            sign=self.level,
            threshold=round(self.parent.threshold, 2),
        )

    @property
    def is_leaf(self):
        """Return whether a node is terminal.

        A decision node is a leaf node if it contains no children.

        Returns
        -------
        bool
        """
        return not self.children
