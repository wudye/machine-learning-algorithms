"""
马尔可夫链的遍历定理（Ergodic Theorem）。通过实际模拟随机游走来统计频率。这更接近 MCMC 采样的真实运行逻辑。
1. 核心原理：遍历定理
遍历定理告诉我们：如果一个马尔可夫链是不可约且非周期的，那么随着时间推移，某个状态出现的频率（时间平均） 会收敛到该状态的平稳概率（空间平均）。
简单说：你在某个地方呆的时间越久，说明那里的概率密度越大。

"""

from bisect import bisect_left

import numpy as np


def get_stationary_distribution(P, start_iter=1000, end_iter=2000, random_state=0):
    """遍历定理求离散有限状态马尔可夫链的某个平稳分布

    要求离散状态、有限状态马尔可夫链是不可约、非周期的。

    :param P: 转移概率矩阵
    :param start_iter: 认为多少次迭代之后状态分布就是平稳分布
    :param end_iter: 计算从start_iter次迭代到end_iter次迭代的状态分布
    :param random_state: 随机种子
    :return: 平稳分布
    """
    n_components = len(P)
    np.random.seed(random_state)

    # 计算累计概率用于随机抽样 根据列随机矩阵视角，将转移概率取出。
    Q = P.T
    for i in range(n_components):
        for j in range(1, n_components):
            Q[i][j] += Q[i][j - 1]

    # 设置初始状态
    x = 0

    # start_iter次迭代
    # 。刚开始的位置是随机的（这里设为 0），需要走一段时间才能进入“平稳区”。这一阶段的记录被丢弃了。
    for _ in range(start_iter):
        v = np.random.rand()
        x = bisect_left(Q[x], v)

    F = np.zeros(n_components)
    # start_iter次迭代到end_iter次迭代
    for _ in range(start_iter, end_iter):
        v = np.random.rand()
        x = bisect_left(Q[x], v)
        F[x] += 1

    return F / sum(F)


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])

    print(get_stationary_distribution(P))  # [0.39 0.18 0.43]

    P = np.array([[1, 1 / 3, 0],
                  [0, 1 / 3, 0],
                  [0, 1 / 3, 1]])

    print(get_stationary_distribution(P))  # [1. 0. 0.]