import numpy as np


class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.activationfunc = self._unitstepfunc_
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        _n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_ = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activationfunc(linear_)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("call fit")

        linear_ = np.dot(X, self.weights) + self.bias
        y_predicted = self.activationfunc(linear_)
        return y_predicted

    def _unitstepfunc_(self, x):
        return np.where(x >= 0, 1, 0)
