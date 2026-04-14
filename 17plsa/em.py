import numpy as np

"""
 概率潜在语义分析 (PLSA, Probabilistic Latent Semantic Analysis) 中的 EM（期望最大化）算法。
PLSA 的核心思想是：一篇文档是由多个“潜在话题（Topic）”混合而成的，而每个话题又是由一组特定的“单词”概率分布构成的。通过 EM 算法，我们可以从纯词频数据中提取出这些看不见的话题。
1. 输入输出与隐变量
输入 X：共现矩阵。行代表 11 个不同的单词，列代表 9 篇具体的文档。矩阵里的数字 <span>X_{ij}</span> 表示第 <span>i</span> 个单词在第 <span>j</span> 篇文档里的出现次数（词频）。
隐含参数：
P1 即 <span>P(w_i|z_k)</span>：表示在给定了某话题 <span>z_k</span> 的情况下，生成某个单词 <span>w_i</span> 的概率。（单词-话题矩阵）
P2 即 <span>P(z_k|d_j)</span>：表示在某篇具体的文档 <span>d_j</span> 中，各个话题 <span>z_k</span> 所占的比例。（话题-文档矩阵）

"""

def em_for_plsa(X, K, max_iter=100, random_state=0):
    """概率潜在语义模型参数估计的EM算法

    :param X: 单词-文本共现矩阵
    :param K: 话题数量
    :param max_iter: 最大迭代次数
    :param random_state: 随机种子
    :return: P(w_i|z_k)和P(z_k|d_j)
    """
    n_features, n_samples = X.shape

    # 计算n(d_j)
    N = [np.sum(X[:, j]) for j in range(n_samples)]

    # 设置参数P(w_i|z_k)和P(z_k|d_j)的初始值
    np.random.seed(random_state)
    P1 = np.random.random((n_features, K))  # P(w_i|z_k)
    P2 = np.random.random((K, n_samples))  # P(z_k|d_j)

    for _ in range(max_iter):
        # E步
        """
        物理意义：面对文档 <span>j</span> 中出现的单词 <span>i</span>，我们去猜：这个单词是由第 <span>k</span> 个话题“贡献”/“生成”的概率有多大？ 这是一个三维张量 P，记录了每个 (单词, 文档, 话题) 组合的贡献度分配。
        """
        P = np.zeros((n_features, n_samples, K))
        for i in range(n_features):
            for j in range(n_samples):
                for k in range(K):
                    P[i][j][k] = P1[i][k] * P2[k][j]
                P[i][j] /= np.sum(P[i][j])

        # M步
        for k in range(K):
            for i in range(n_features):
                P1[i][k] = np.sum([X[i][j] * P[i][j][k] for j in range(n_samples)])
            P1[:, k] /= np.sum(P1[:, k])

        for k in range(K):
            for j in range(n_samples):
                P2[k][j] = np.sum([X[i][j] * P[i][j][k] for i in range(n_features)]) / N[j]

    return P1, P2


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 2, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 0]])

    np.set_printoptions(precision=2, suppress=True)
    R1, R2 = em_for_plsa(X, 3)

    print(R1)
    # [[0.   0.15 0.  ]
    #  [0.15 0.   0.  ]
    #  [0.   0.   0.4 ]
    #  [0.15 0.   0.  ]
    #  [0.08 0.08 0.  ]
    #  [0.23 0.31 0.4 ]
    #  [0.   0.15 0.  ]
    #  [0.15 0.   0.  ]
    #  [0.23 0.   0.  ]
    #  [0.   0.15 0.2 ]
    #  [0.   0.15 0.  ]]

    print(R2)
    # [[0. 0. 0. 0. 0. 1. 1. 0. 1.]
    #  [1. 0. 1. 1. 1. 0. 0. 0. 0.]
    #  [0. 1. 0. 0. 0. 0. 0. 1. 0.]]