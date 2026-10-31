from collections import Counter
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def euclidean_distance(x1: npt.NDArray, x2: npt.NDArray) -> float:
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN[T]:
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, X: npt.NDArray[np.float64], y: Sequence[T]) -> None:
        self.X_train = X
        self.y_train = y

    def predict(self, X: npt.NDArray[np.float64]) -> list[T]:
        return [self._predict(x) for x in X]

    def _predict(self, x: npt.NDArray) -> T:
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
