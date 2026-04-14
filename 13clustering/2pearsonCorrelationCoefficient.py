
import numpy as np


def correlation_coefficient(X):

    n = len(X[0])
    #     # np.mean(X, axis=0) 沿着列的方向求均值。
    means = np.mean(X, axis=0)
    means_c = np.mean(X,axis=1)
    print(means, means_c)

    variance = [np.square(X[:, i] - means[i]).sum() for i in range(n)]

    print(variance)
    D = np.identity(n)

    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = X[:, i], X[:, j]
            numerator = ((xi - means[i]) * (xj - means[j])).sum()
            denominator = np.sqrt(variance[i] * variance[j])
            if denominator:
                D[i][j] = D[j][i] = numerator / denominator
            else:
                D[i][j] = D[j][i] = np.nan
    return D


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 5, 5],
                  [2, 0, 0, 0, 2]])

    print(correlation_coefficient(X))