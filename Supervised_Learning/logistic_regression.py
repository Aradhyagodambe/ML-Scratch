import numpy as np
import numpy.typing as npt


class LogisticRegression:
    def __init__(self, lr: float = 0.001, n_iters: int = 1000) -> None:
        self.lr: float = lr
        self.n_iters: int = n_iters
        self.weights: npt.NDArray[np.float64] | None = None
        self.bias: float | None = None

    def fit(
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_ | np.float64]
    ) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw: npt.NDArray[np.float64] = (1 / n_samples) * np.dot(
                X.T, (y_predicted - y)
            )
            db: float = float(1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: npt.NDArray[np.float64]):
        if self.weights is None or self.bias is None:
            raise ValueError("call fit")
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls: list[int] = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x: npt.NDArray[np.float64]):
        # Bound values to prevent overflow in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
