import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y, *_ = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4, coef=False
)

X_arr: npt.NDArray[np.float64] = np.asarray(X)
y_arr: npt.NDArray[np.float64] = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

X_train_arr: npt.NDArray[np.float64] = np.asarray(X_train)
X_test_arr: npt.NDArray[np.float64] = np.asarray(X_test)
y_train_arr: npt.NDArray[np.float64] = np.asarray(y_train)
y_test_arr: npt.NDArray[np.float64] = np.asarray(y_test)

from LinearRegression_Scratch import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train_arr, y_train_arr)
predicted: npt.NDArray[np.float64] = regressor.predict(X_test_arr)


def mse(y_true: npt.NDArray, y_predicted:npt.NDArray):
    return np.mean((y_true - y_predicted) ** 2)


mse_value = mse(y_test_arr, predicted)
print(mse_value)

y_pred_line: npt.NDArray = regressor.predict(X_arr)

cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train_arr, y_train_arr, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test_arr, y_test_arr, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="prediction")
plt.show()
