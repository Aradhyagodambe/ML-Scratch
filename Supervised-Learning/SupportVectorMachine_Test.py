import matplotlib.pylab as plt
import numpy as np
from sklearn import datasets

from SupportVectorMachine import SupportVectorMachine

blobs = datasets.make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
)
X: np.ndarray = np.asarray(blobs[0])
y: np.ndarray = np.asarray(blobs[1])

y = np.where(y == 0, -1, 1)

clf = SupportVectorMachine()
clf.fit(X, y)

print(clf.weight, clf.bias)


def visualize_():
    def hyperplane_(x, w, b, offset):
        return float(-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x0_1 = float(np.amin(X[:, 0]))
    x0_2 = float(np.amax(X[:, 0]))

    x1_1 = hyperplane_(x0_1, clf.weight, clf.bias, 0)
    x1_2 = hyperplane_(x0_2, clf.weight, clf.bias, 0)

    x1_1_m = hyperplane_(x0_1, clf.weight, clf.bias, -1)
    x1_2_m = hyperplane_(x0_2, clf.weight, clf.bias, -1)

    x1_1_p = hyperplane_(x0_1, clf.weight, clf.bias, 1)
    x1_2_p = hyperplane_(x0_2, clf.weight, clf.bias, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = float(np.amin(X[:, 1]))
    x1_max = float(np.amax(X[:, 1]))

    ax.set_ylim((x1_min - 3, x1_max + 3))

    plt.show()


visualize_()
