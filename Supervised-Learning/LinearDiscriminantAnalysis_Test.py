import matplotlib.pyplot as plt
from sklearn import datasets

from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis

X, y = datasets.load_iris(return_X_y=True)

lda = LinearDiscriminantAnalysis(2)
lda.fit(X, y)
X_projected = lda.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2, c=y, edgecolors="none", alpha=0.8, cmap="viridis")

plt.xlabel("linear Discriminant 1")
plt.ylabel("linear Discriminant 2")
plt.colorbar()
plt.show()
