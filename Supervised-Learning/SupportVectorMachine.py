import numpy as np


class SupportVectorMachine:
    def __init__(self, lr=0.001, lambda_=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        _n_samples, n_features = X.shape

        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                cond = y_[idx] * (np.dot(x_i, self.weight) - self.bias) >= 1
                if cond:
                    self.weight -= self.lr * (2 * self.lambda_ * self.weight)
                else:
                    self.weight -= self.lr * (
                        2 * self.lambda_ * self.weight - x_i * y_[idx]
                    )
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        if self.weight is None or self.bias is None:
            return ValueError("call fit")
        linear_ = np.dot(X, self.weight) - self.bias
        return np.sign(linear_)
