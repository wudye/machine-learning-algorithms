import numpy as np
# 散布矩阵 <span>A</span>（也叫离差阵）： 衡量的是数据去均值后的外积之和
def scatter_matrix(X, G):
    """计算样本散布矩阵

    :param X: 样本
    :param G: 类别包含的样本
    :return: 样本散步矩阵
    """
    n_samples = len(G)
    n_features = len(X)

    # 计算类的中心
    means = np.mean(X[:, G], axis=1)
    print(means)

    A = np.zeros((n_features, n_features))
    """
    在 NumPy 中，切片 X[:, i] 提取出来的是一个一维数组，形状（shape）是 (2,)。
    使用 np.newaxis 的核心目的是将一维数组（一维向量）提升为二维的列向量（矩阵）
    """
    for i in G:
        A += np.dot((X[:, i] - means)[:, np.newaxis], (X[:, i] - means)[:, np.newaxis].T)


    return A

"""
协方差矩阵 <span>S</span>： 散布矩阵受限于样本数量，样本越多，散布矩阵里的数值越大。
为了消除样本数量的影响，我们需要对其求平均，即除以自由度。对于无偏样本协方差矩阵
"""
def covariance_matrix(X, G):
    """计算样本协方差矩阵"""
    n_features = len(G)
    A = scatter_matrix(X, G)
    # 除以无偏估计的自由度 (样本数 - 1)
    if n_features > 1:
        S = A / (n_features - 1)
    else:
        # 当只有一个样本时，协方差矩阵全为0（或不存在方差），防止出现除以0的报错
        S = np.zeros_like(A)
    return S

if __name__ == "__main__":
    X = np.array([[0, 0, 1, 5, 5],
                  [2, 0, 0, 0, 2]])



    print(scatter_matrix(X, [2, 3, 4]))


    print(covariance_matrix(X, [2, 3, 4]))
