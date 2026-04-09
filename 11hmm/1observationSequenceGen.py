
from bisect import bisect_left
from copy import deepcopy
from random import random


def build_markov_sequence(a, b, p, t):
    a = deepcopy(a)
    for i in range(len(a)):
        for j in range(1, len(a[0])):
            a[i][j] += a[i][j - 1]
    print(a)
    b = deepcopy(b)
    for i in range(len(b)):
        for j in range(1, len(b[0])):
            b[i][j] += b[i][j - 1]
    print(b)
    p = deepcopy(p)
    for i in range(1, len(p)):
        p[i] += p[i - 1]
    print(p)
    # 确定初始的隐藏状态（选盒子）
    stat = [bisect_left(p, random())]
    # 基于该状态生成初始观测值（从刚选的盒子里摸球）
    res = [bisect_left(b[stat[-1]], random())]

    for _ in range(1,t):
        # 状态转移（从当前盒子走向下一个盒子）
        stat.append(bisect_left(a[stat[-1]], random()))
        # 生成新的观测值（从新的盒子里抽出球）
        res.append(bisect_left(b[stat[-1]], random()))
    return res



if __name__ == "__main__":
    # 状态转移概率矩阵, 比如 a[i][j] 表示从隐藏状态 i 转移到隐藏状态 j 的概率。
    A = [[0.0, 1.0, 0.0, 0.0],
         [0.4, 0.0, 0.6, 0.0],
         [0.0, 0.4, 0.0, 0.6],
         [0.0, 0.0, 0.5, 0.5]]
    # 观测概率矩阵。比如 b[i][k] 表示在隐藏状态 i 下，生成观测值 k 的概率。
    B = [[0.5, 0.5],
         [0.3, 0.7],
         [0.6, 0.4],
         [0.8, 0.2]]
    # 初始状态概率分布
    p = [0.25, 0.25, 0.25, 0.25]
    print(build_markov_sequence(A, B, p, 200))

