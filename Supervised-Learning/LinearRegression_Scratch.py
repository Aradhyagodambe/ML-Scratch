import numpy as np
import numpy.typing as npt


class LinearRegression:
    def __init__(self, lr: float = 0.01, n_iters: int = 1000) -> None:
        self.lr: float = lr
        self.n_iters: int = n_iters
        self.weights: npt.NDArray | None = None
        self.bias: float | None = None

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw: npt.NDArray = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db: float = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        if self.weights is None or self.bias is None:
            raise ValueError("call fit")
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
