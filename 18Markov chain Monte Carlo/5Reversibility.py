"""
判断一个马尔可夫链是否满足可逆性（Reversibility），也就是是否满足细致平衡方程（Detailed Balance Equation）。
在 MCMC 采样中，可逆性是保证马尔可夫链能收敛到目标平稳分布的关键前提。
核心数学原理：细致平衡
一个马尔可夫链如果可逆，必须存在一个概率分布 a（平稳分布），使得对于任意两个状态 i和j
，都满足：
a[i] * P[i][j] = a[j] * P[j][i]
其中 P[i][j] 是从状态 i 转移到状态 j 的概率。
如果上述方程对于所有状态对 (i, j) 都成立，那么这个马尔可夫链就是可逆的，并且 a 就是它的平稳分布。
"""

import numpy as np


def get_stationary_distribution(P, tol=1e-8, max_iter=1000):
    """迭代法求离散有限状态马尔可夫链的某个平稳分布

    根据平稳分布的定义求平稳分布。如果有无穷多个平稳分布，则返回其中任意一个。如果不存在平稳分布，则无法收敛。

    :param P: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 平稳分布
    """
    n_components = len(P)

    # 初始状态分布：均匀分布
    pi0 = np.array([1 / n_components] * n_components)

    # 迭代寻找平稳状态
    for _ in range(max_iter):
        pi1 = np.dot(P, pi0)

        # 判断迭代更新量是否小于容差
        if np.sum(np.abs(pi0 - pi1)) < tol:
            break
        # print(pi0, pi1)
        pi0 = pi1

    return pi0


def is_reversible(P, tol=1e-4, max_iter=1000):
    """计算有限状态马尔可夫链是否可逆

    :param P: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 可逆 = True ; 不可逆 = False
    """
    n_components = len(P)
    D = get_stationary_distribution(P, pow(tol, 2), max_iter)  # 计算平稳分布
    for i in range(n_components):
        for j in range(n_components):
            if not - tol < P[i][j] * D[j] - P[j][i] * D[i] < tol:
                return False
    return True


if __name__ == "__main__":
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])

    print(is_reversible(P))  # True

    P = np.array([[0.25, 0.5, 0.25],
                  [0.25, 0, 0.5],
                  [0.5, 0.5, 0.25]])

    print(is_reversible(P))  # False