from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def entropy(y: npt.NDArray[np.int_]) -> float:
    ps = np.bincount(y) / len(y)
    ps = ps[ps > 0]
    return float(-np.sum(ps * np.log2(ps)))


@dataclass(frozen=True)
class Leaf:
    value: int


@dataclass(frozen=True)
class Split:
    feature: int
    threshold: float
    left: Node
    right: Node


type Node = Leaf | Split


class DecisionTree:
    def __init__(
        self,
        min_samples_split: int = 2,
        max_depth: int = 100,
        n_feats: int | None = None,
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats  
        self.root: Node | None = None
        self._n_feats: int = 0 

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_]) -> None:
        n_cols = X.shape[1]
        self._n_feats = n_cols if self.n_feats is None else min(self.n_feats, n_cols)
        self.root = self._grow_tree(X, y)

    def _grow_tree(
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_], depth: int = 0
    ) -> Node:
        n_samples, n_features = X.shape

        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or n_samples < self.min_samples_split
        ):
            return Leaf(self._most_common_label(y))

        feat_idxs = np.random.choice(n_features, self._n_feats, replace=False)
        best = self._best_criteria(X, y, feat_idxs)

        if best is None:
            return Leaf(self._most_common_label(y))

        best_feat, best_thresh = best
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        return Split(
            best_feat,
            best_thresh,
            self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1),
            self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1),
        )

    def _best_criteria(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int_],
        feat_idxs: npt.NDArray[np.int_],
    ) -> tuple[int, float] | None:  
        best_gain = 0.0
        best: tuple[int, float] | None = None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            for threshold in np.unique(X_column):
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best = (int(feat_idx), float(threshold))

        return best

    def _information_gain(
        self,
        y: npt.NDArray[np.int_],
        X_column: npt.NDArray[np.float64],
        split_thresh: float,
    ) -> float:
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0  

        n = len(y)
        child_entropy = (len(left_idxs) / n) * entropy(y[left_idxs]) + (
            len(right_idxs) / n
        ) * entropy(y[right_idxs])

        return entropy(y) - child_entropy

    def _split(
        self, X_column: npt.NDArray[np.float64], split_thresh: float
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
        if self.root is None:
            raise ValueError("call fit")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: npt.NDArray[np.float64], node: Node) -> int:
        if isinstance(node, Leaf):
            return node.value
            
        child = node.left if x[node.feature] <= node.threshold else node.right
        return self._traverse_tree(x, child)

    def _most_common_label(self, y: npt.NDArray[np.int_]) -> int:
        return int(Counter(y).most_common(1)[0][0])  