import numpy as np
from KNN_Scratch import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

X_train_arr = np.asarray(X_train, dtype=np.float64)
X_test_arr = np.asarray(X_test, dtype=np.float64)
y_train_list = [int(label) for label in y_train]

clf = KNN[int](k=3)
clf.fit(X_train_arr, y_train_list)

predictions = clf.predict(X_test_arr)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
