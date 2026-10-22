import numpy as np


class PrincipalComponentAnalysis:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.components: np.ndarray | None = None
        self.mean: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X: np.ndarray):
        if self.components is None or self.mean is None:
            raise RuntimeError("call fit")
        X = X - self.mean
        return np.dot(X, self.components.T)
