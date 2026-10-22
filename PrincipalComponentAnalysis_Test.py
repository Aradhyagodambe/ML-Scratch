import matplotlib.pyplot as plt
from sklearn import datasets

from PrincipalComponentAnalysis import PrincipalComponentAnalysis

X, y = datasets.load_iris(return_X_y=True)


PCA = PrincipalComponentAnalysis(2)
PCA.fit(X)
X_projected = PCA.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(
    x1, x2, c=y, edgecolors="none", alpha=0.8, cmap="viridis"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()
