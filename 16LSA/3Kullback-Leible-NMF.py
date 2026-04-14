import numpy as np


def nmp_training(X, k, max_iter=100, tol=1e-4, random_state=0):
    """非负矩阵分解的迭代算法（散度）

    :param X: 单词-文本矩阵
    :param k: 文本集合的话题个数k
    :param max_iter: 最大迭代次数
    :param tol: 容差
    :param random_state: 随机种子
    :return: 话题矩阵W,文本表示矩阵H
    """
    n_features, n_samples = X.shape

    # 初始化
    np.random.seed(random_state)
    W = np.random.random((n_features, k))
    H = np.random.random((k, n_samples))

    def score():
        """计算散度的损失函数
         广义 Kullback-Leibler (KL) 散度（也称为相对熵），它是非负矩阵分解（NMF）中除了平方误差（欧氏距离）之外，另一种非常经典的损失函数（Loss Function）。
        """
        Y = np.dot(W, H)
        score = 0
        for i in range(n_features):
            for j in range(n_samples):
                score += (X[i][j] * np.log(X[i][j] / Y[i][j]) if X[i][j] * Y[i][j] > 0 else 0) + (- X[i][j] + Y[i][j])
        return score

    # 计算当前损失函数
    last_score = score()

    # 迭代
    """
    在 NMF 中，除了欧氏距离，最常用的就是广义 KL 散度。 每次迭代都会计算当前预测矩阵
     <span>Y = WH</span> 与真实矩阵 <span>X</span> 的差异。散度越小，说明 <span>W \times H</span> 越逼近 <span>X</span>。
    """
    # (单词-话题矩阵)：每一行对应一个单词，每一列对应一个话题。元素 W[i][l] 表示第 i 个单词在第 l 个话题中的权重。
    for _ in range(max_iter):
        # 更新W的元素
        WH = np.dot(W, H)
        for i in range(n_features):
            for l in range(k):
                v1 = sum(H[l][j] * X[i][j] / WH[i][j] for j in range(n_samples))
                v2 = sum(H[l][j] for j in range(n_samples))
                W[i][l] *= v1 / v2

        # 更新H的元素
        # (话题-文本矩阵)：每一行对应一个话题，每一列对应一个文本。元素 H[l][j] 表示第 l 个话题在第 j 个文本中的权重。
        WH = np.dot(W, H)
        for l in range(k):
            for j in range(n_samples):
                v1 = sum(W[i][l] * X[i][j] / WH[i][j] for i in range(n_features))
                v2 = sum(W[i][l] for i in range(n_features))
                H[l][j] *= v1 / v2

        now_score = score()
        if last_score - now_score < tol:
            break

        last_score = now_score

    return W, H


if __name__ == "__main__":
    """
    NMF 的目标是将一个非负的原始矩阵 <span>X</span>（通常是单词-文本的词频矩阵）
    ，近似分解为两个低秩的非负矩阵 <span>W</span>（单词-话题矩阵）和
     <span>H</span>（话题-文本矩阵）的乘积，即 <span>X prox W \times H</span>。
         这是一种无监督的聚类/降维方法，广泛用于文本的主题提取（Topic Modeling）"""
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
    W, H = nmp_training(X, 3)

    print(W)
    # [[0.   0.42 0.  ]
    #  [0.35 0.   0.  ]
    #  [0.   0.   0.91]
    #  [0.35 0.   0.  ]
    #  [0.18 0.21 0.  ]
    #  [0.53 0.85 0.91]
    #  [0.   0.42 0.  ]
    #  [0.35 0.   0.  ]
    #  [0.53 0.   0.  ]
    #  [0.   0.42 0.45]
    #  [0.   0.42 0.  ]]

    print(H)
    # [[0.   0.   0.   0.   0.   2.19 1.31 0.   2.19]
    #  [1.45 0.   1.45 1.09 0.72 0.   0.   0.   0.  ]
    #  [0.   0.88 0.   0.   0.   0.   0.   1.32 0.  ]]

    Y = np.dot(W, H)
    print(Y)
    # [[0.62 0.   0.62 0.46 0.31 0.   0.   0.   0.  ]
    #  [0.   0.   0.   0.   0.   0.77 0.46 0.   0.77]
    #  [0.   0.8  0.   0.   0.   0.   0.   1.2  0.  ]
    #  [0.   0.   0.   0.   0.   0.77 0.46 0.   0.77]
    #  [0.31 0.   0.31 0.23 0.15 0.38 0.23 0.   0.38]
    #  [1.23 0.8  1.23 0.92 0.62 1.15 0.69 1.2  1.15]
    #  [0.62 0.   0.62 0.46 0.31 0.   0.   0.   0.  ]
    #  [0.   0.   0.   0.   0.   0.77 0.46 0.   0.77]
    #  [0.   0.   0.   0.   0.   1.15 0.69 0.   1.15]
    #  [0.62 0.4  0.62 0.46 0.31 0.   0.   0.6  0.  ]
    #  [0.62 0.   0.62 0.46 0.31 0.   0.   0.   0.  ]]

    score = 0
    s1, s2 = X.shape
    for i in range(s1):
        for j in range(s2):
            score += (X[i][j] * np.log(X[i][j] / Y[i][j]) if X[i][j] * Y[i][j] > 0 else 0) + (- X[i][j] + Y[i][j])
    print(score)  # 11.664003859511485