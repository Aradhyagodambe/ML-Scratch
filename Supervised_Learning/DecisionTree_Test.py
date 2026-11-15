import numpy as np
import numpy.typing as npt
from DecisionTree import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split


def accuracy(y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_]) -> float:
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

X_train_arr: npt.NDArray[np.float64] = np.asanyarray(X_train)
X_test_arr: npt.NDArray[np.float64] = np.asanyarray(X_test)
y_train_arr: npt.NDArray[np.int_] = np.asanyarray(y_train)
y_test_arr: npt.NDArray[np.int_] = np.asanyarray(y_test)

clf = DecisionTree(max_depth=10)
clf.fit(X_train_arr, y_train_arr)

y_pred = clf.predict(X_test_arr)
acc = accuracy(y_test_arr, y_pred)
print("accuracy:", acc)
