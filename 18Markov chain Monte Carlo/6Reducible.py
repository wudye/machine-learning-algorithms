"""
在 MCMC 采样中，我们希望马尔可夫链是不可约（Irreducible）的，这意味着从任意一个状态出发，经过有限步迭代，最终都能到达其他任何一个状态。
1. 核心数学概念
不可约（Irreducible）：整个状态空间是一个“强连通”的整体。没有孤岛，你可以从任何地方去往任何地方。
可约（Reducible）：状态空间可以被拆分成几个独立的子集。一旦进入某个子集，就再也回不去其他地方了（或者永远进不去某些地方）。

"""

import numpy as np


def is_reducible(P):
    """计算马尔可夫链是否可约

    :param P: 转移概率矩阵
    :return: 可约 = True ; 不可约 = False
    """
    n_components = len(P)

    # 遍历所有状态k，检查从状态k出发能否到达任意状态
    for k in range(n_components):
        visited = set()  # 当前已遍历过的状态

        find = False  # 当前是否已找到可到达任意位置的时刻
        # 初始状态：假设当前只在状态 k （stat0 中只有索引 k 为 True）。
        stat0 = (False,) * k + (True,) + (False,) * (n_components - k - 1)  # 时刻0可达到的位置
        print((False,) * k ,  k, (False,) * (n_components - k - 1), stat0)

        while stat0 not in visited:
            visited.add(stat0)
            stat1 = [False] * n_components

            for j in range(n_components):
                if stat0[j] is True:
                    for i in range(n_components):
                        if P[i][j] > 0:
                            stat1[i] = True

            # 如果已经到达之前已检查可到达任意状态的状态，则不再继续寻找
            for i in range(k):
                if stat1[i] is True:
                    find = True
                    break

            # 如果当前时刻可到达任意位置，则不再寻找
            if all(stat1) is True:
                find = True
                break

            stat0 = tuple(stat1)

        if not find:
            return True

    return False


if __name__ == "__main__":
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])

    print(is_reducible(P))  # False

    """
    观察第三列 [0, 0, 1]。这表示：
    如果你在状态 2，你转移到状态 0 的概率是 0，转移到状态 1 的概率也是 0，你只能留在状态 2。
    状态 2 变成了一个“陷阱”或“孤岛”。你进得来（从状态 1 有 0.5 的概率进来），但你出不去。
    """
    P = np.array([[0, 0.5, 0],
                  [1, 0, 0],
                  [0, 0.5, 1]])

    print(is_reducible(P))  # True