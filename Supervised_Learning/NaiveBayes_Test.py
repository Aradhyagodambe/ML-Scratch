import numpy as np
import numpy.typing as npt
from NaiveBayes import NaiveBayes
from sklearn import datasets
from sklearn.model_selection import train_test_split


def accuracy(y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_]) -> float:
    return float(np.sum(y_true == y_pred) / len(y_true))


X, y = datasets.make_classification(
    n_samples=100, n_features=10, n_classes=2, random_state=123
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

X_train_arr: npt.NDArray[np.float64] = np.asarray(X_train)
X_test_arr: npt.NDArray[np.float64] = np.asarray(X_test)
y_train_arr: npt.NDArray[np.int_] = np.asarray(y_train)
y_test_arr: npt.NDArray[np.int_] = np.asarray(y_test)

nb = NaiveBayes()
nb.fit(X_train_arr, y_train_arr)
predictions = nb.predict(X_test_arr)

print("accuracy:", accuracy(y_test_arr, predictions))
