import numpy as np


def get_word_document_matrix(D):
    """依据TF-IDF构造的单词-文本矩阵
    TF = 单词出现在文本中的频数/文本中出现的所有单词的频数之和
    IDF = 包含单词的文本数/文本集合D的全部文本数
    乘积TF-IDF = TF * IDF
    :param D: 文本集合
    :return: 依据TF-IDF的单词-文本矩阵
    """
    n_samples = len(D)

    # 构造所有文本出现的单词的集合
    W = set()
    for d in D:
        W |= set(d)

    # 构造单词列表及单词下标映射
    W = sorted(W)
    mapping = {w: i for i, w in enumerate(W)}

    n_features = len(W)

    # 计算：单词出现在文本中的频数/文本中出现的所有单词的频数之和
    X = np.zeros((n_features, n_samples))
    for i, d in enumerate(D):
        for w in d:
            X[mapping[w], i] += 1
        #print("文本：", d, "文本下标：", i, "单词频数：", X[:, i])
        X[:, i] /= len(d)
        #print(X[:, i])

    # 计算：包含单词的文本数/文本集合D的全部文本数
    """
    加了 set(d) 变成集合，强制去重。因为**“文档频率（DF）”只关心这篇文档里
    “有”还是“没有”这个词，而不关心它在这篇特定文档里出现了多少次**
    （出现次数这是前面 TF 词频关心的事情）。无论 "rich" 在这篇文档里出现 1 
    次还是 100 次，对于计算“包含该词的文档总数”来说，这篇文档都只能算作 1 次。
    """
    df = np.zeros(n_features)
    for d in D:
        for w in set(d):
            df[mapping[w]] += 1
    print("df", df)

    # 构造单词-文本矩阵
    """
    计算每个单词的逆文档频率（IDF），并将其与之前算好的词频（TF）相乘，
    最终得到完整的 TF-IDF 权重矩阵
    IDF 公式（惩罚常见词，奖励罕见词）
    极度常见的词：比如在你的测试用例里，"investing" 这个词在所有 9 篇文档中都出现了。此时 df[i] = 9，n_samples = 9。那么 n_samples / df[i] = 1，而 <span>\log(1) = 0</span>。这意味着这个词的 IDF 权重会被打成 0（毫无区分度）。
    比较罕见的词：如果某个词只在 1 篇文档里出现过，那 9 / 1 = 9，<span>\log(9)</span> 就会得到一个较大的正数权重。
    为什么要加 <span>\log</span>：如果不加对数，罕见词的权重会呈线性爆炸式增长。套上 <span>\log</span> 可以平滑这种极端影响。
    """
    for i in range(n_features):
        X[i, :] *= np.log(n_samples / df[i])

    return X


if __name__ == "__main__":
    D = [["guide", "investing", "market", "stock"],
         ["dummies", "investing"],
         ["book", "investing", "market", "stock"],
         ["book", "investing", "value"],
         ["investing", "value"],
         ["dads", "guide", "investing", "rich", "rich"],
         ["estate", "investing", "real"],
         ["dummies", "investing", "stock"],
         ["dads", "estate", "investing", "real", "rich"]]

    np.set_printoptions(precision=2)
    print(get_word_document_matrix(D))

    # [[0.   0.   0.38 0.5  0.   0.   0.   0.   0.  ]
    #  [0.   0.   0.   0.   0.   0.3  0.   0.   0.3 ]
    #  [0.   0.75 0.   0.   0.   0.   0.   0.5  0.  ]
    #  [0.   0.   0.   0.   0.   0.   0.5  0.   0.3 ]
    #  [0.38 0.   0.   0.   0.   0.3  0.   0.   0.  ]
    #  [0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
    #  [0.38 0.   0.38 0.   0.   0.   0.   0.   0.  ]
    #  [0.   0.   0.   0.   0.   0.   0.5  0.   0.3 ]
    #  [0.   0.   0.   0.   0.   0.6  0.   0.   0.3 ]
    #  [0.27 0.   0.27 0.   0.   0.   0.   0.37 0.  ]
    #  [0.   0.   0.   0.5  0.75 0.   0.   0.   0.  ]]