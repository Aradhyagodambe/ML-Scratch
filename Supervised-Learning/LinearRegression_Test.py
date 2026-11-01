import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn import datasets
from sklearn.model_selection import train_test_split

X_raw, y_raw, *_ = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4, coef=False
)

X: npt.NDArray = np.asarray(X_raw)
y: npt.NDArray = np.asarray(y_raw)



X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

X_train:npt.NDArray = np.asarray(X_train_raw)
X_test:npt.NDArray = np.asarray(X_test_raw)
y_train:npt.NDArray =  np.asarray(y_train_raw)
y_test:npt.NDArray = np.asarray(y_test_raw)

from LinearRegression_Scratch import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
predicted: np.ndarray = regressor.predict(X_test)


def mse(y_true: npt.NDArray, y_predicted:npt.NDArray):
    return np.mean((y_true - y_predicted) ** 2)


mse_value = mse(y_test, predicted)
print(mse_value)

y_pred_line: npt.NDArray = regressor.predict(X)

cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="prediction")
plt.show()
