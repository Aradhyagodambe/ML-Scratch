import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from LogisticRegression import LogisticRegression

X, y = datasets.load_breast_cancer(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


regressor = LogisticRegression(lr=0.01)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("accuracy:", accuracy(y_test, predictions))
