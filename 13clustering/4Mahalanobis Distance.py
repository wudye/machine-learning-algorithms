
import numpy as np


def mahalanobis_distance(X):
    n_samples = len(X[0])

    # 计算协方差矩阵
    S = np.cov(X)
    print("Covariance Matrix:\n", S)

    # 计算协方差矩阵的逆矩阵
    S = np.linalg.inv(S)
    print("Inverse Covariance Matrix:\n", S)

    # 构造马哈拉诺比斯距离矩阵
    D = np.zeros((n_samples, n_samples))  # 初始化为零矩阵

    """
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            xi = X[:, i][:, np.newaxis]
            xj = X[:, j][:, np.newaxis]
            D[i][j] = D[j][i] = np.sqrt((np.dot(np.dot((xi - xj).T, S), (xi - xj)))).item()

    """

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # 取出一维向量，形状为 (2,)
            xi = X[:, i]
            xj = X[:, j]

            # 维向量直接用点积计算，np.dot(一维, 一维) 会直接返回标量数字
            D[i][j] = D[j][i] = np.sqrt(np.dot(np.dot(xi - xj, S), xi - xj))

    return D



if __name__ == "__main__":
    X = np.array([[0, 0, 1, 5, 5],
                  [2, 0, 0, 0, 2]])
    print(mahalanobis_distance(X))