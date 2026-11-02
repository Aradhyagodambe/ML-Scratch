import numpy as np
import numpy.typing as npt
from LogisticRegression import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_breast_cancer(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

X_train_arr = np.asarray(X_train, dtype=np.float64)
X_test_arr = np.asarray(X_test, dtype=np.float64)
y_train_arr = np.asarray(y_train, dtype=np.int_)
y_test_arr = np.asarray(y_test, dtype=np.int_)

def accuracy(y_true: npt.NDArray[np.int_], y_pred:list[int] |npt.NDArray[np.int_]) -> float:
    accuracy: float = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


regressor = LogisticRegression(lr=0.01)
regressor.fit(X_train_arr, y_train_arr)
predictions: list[int] = regressor.predict(X_test_arr)

print("accuracy:", accuracy(y_test_arr, predictions))
