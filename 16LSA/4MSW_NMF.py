import numpy as np

"""
在非负矩阵分解（NMF）中，WH（即代码里的 np.dot(W, H)）是模型对原始数据的“预测矩阵”或“重构矩阵”。
1. 数学层面的含义
NMF 的根本目标就是找两个较小的矩阵 <span>W</span> 和 <span>H</span>，使得它们的乘积尽可能等于原始矩阵 <span>X</span>： <span>X \approx W \times H</span> 所以，<span>WH</span> 就是当前这一轮迭代中，模型拼凑出来的近似值。我们在衡量目前模型训练得好不好的时候，就是计算真实的 <span>X</span> 和预测的 <span>WH</span> 之间的差异（例如代码中的 np.sum(np.square(X - np.dot(W, H)))，平方损失）。
2. 物理层面的含义（话题模型 / LSA的视角）
<span>W</span> (单词-话题矩阵)：记录了每一个单词在各个“潜在话题”中的重要程度。
<span>H</span> (话题-文本矩阵)：记录了各个“潜在话题”在每一篇文档中占的比例。
<span>WH</span> (预测的单词-文本矩阵)：当你把 <span>W</span> 和 <span>H</span> 乘起来时，相当于利用提取出来的“潜在话题”作为桥梁，重新逆向推导出了每个单词在每篇文档里应该出现的频率。
"""

def nmp_training(X, k, max_iter=100, tol=1e-4, random_state=0):
    """非负矩阵分解的迭代算法（平方损失）

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

    # 计算当前平方损失
    last_score = np.sum(np.square(X - np.dot(W, H)))

    # 迭代
    for _ in range(max_iter):
        # 更新W的元素
        A = np.dot(X, H.T)  # X H^T
        B = np.dot(np.dot(W, H), H.T)  # W H H^T
        for i in range(n_features):
            for l in range(k):
                W[i][l] *= A[i][l] / B[i][l]

        # 更新H的元素
        C = np.dot(W.T, X)  # W^T X
        D = np.dot(np.dot(W.T, W), H)  # W^T W H
        for l in range(k):
            for j in range(n_samples):
                H[l][j] *= C[l][j] / D[l][j]

        # 检查迭代更新量是否已小于容差
        now_score = np.sum(np.square(X - np.dot(W, H)))
        print(last_score - now_score)
        if last_score - now_score < tol:
            break

        last_score = now_score

    return W, H


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
    W, H = nmp_training(X, 3)

    print(W)
    # [[0.   0.26 0.22]
    #  [0.4  0.   0.11]
    #  [0.   0.2  0.19]
    #  [0.08 0.   0.66]
    #  [0.27 0.22 0.  ]
    #  [0.26 0.61 1.22]
    #  [0.   0.46 0.  ]
    #  [0.08 0.   0.66]
    #  [0.69 0.   0.  ]
    #  [0.   0.62 0.  ]
    #  [0.   0.   0.55]]

    print(H)
    # [[0.22 0.   0.   0.   0.   2.88 0.03 0.   1.49]
    #  [1.65 0.42 1.72 0.29 0.   0.17 0.   1.14 0.  ]
    #  [0.   0.39 0.02 0.64 0.64 0.   0.92 0.17 0.7 ]]

    Y = np.dot(W, H)
    print(Y)
    # [[0.43 0.19 0.45 0.21 0.14 0.04 0.2  0.33 0.15]
    #  [0.09 0.04 0.   0.07 0.07 1.16 0.11 0.02 0.68]
    #  [0.33 0.16 0.34 0.18 0.12 0.03 0.18 0.26 0.14]
    #  [0.02 0.25 0.01 0.42 0.42 0.23 0.6  0.11 0.58]
    #  [0.42 0.09 0.38 0.06 0.   0.83 0.01 0.25 0.41]
    #  [1.07 0.73 1.07 0.95 0.79 0.86 1.12 0.91 1.25]
    #  [0.77 0.2  0.8  0.13 0.   0.08 0.   0.53 0.  ]
    #  [0.02 0.25 0.01 0.42 0.42 0.23 0.6  0.11 0.58]
    #  [0.15 0.   0.   0.   0.   1.98 0.02 0.   1.02]
    #  [1.03 0.26 1.06 0.18 0.   0.11 0.   0.7  0.  ]
    #  [0.   0.21 0.01 0.35 0.35 0.   0.5  0.09 0.39]]

    print(np.sum(np.square(X - Y)))  # 7.661276178695074