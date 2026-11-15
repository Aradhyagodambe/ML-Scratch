import numpy as np
import numpy.typing as npt


class NaiveBayes:
    def __init__(self) -> None:
        self._classes: npt.NDArray[np.int_]
        self._mean: npt.NDArray[np.float64]
        self._var: npt.NDArray[np.float64]
        self._priors: npt.NDArray[np.float64]

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_]) -> None:
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[c == y]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred, dtype=np.int_)

    def _predict(self, x: npt.NDArray[np.float64]) -> np.int_:
        posteriors = []

        for idx, _ in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(
        self, class_idx: int, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        mean = self._mean[class_idx]
        var = self._var[class_idx] + 1e-9
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
