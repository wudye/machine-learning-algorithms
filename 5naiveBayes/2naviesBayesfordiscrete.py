import collections


class NaiveBayesAlgorithmArray:
    """朴素贝叶斯算法（仅支持离散型数据）

    使用列表存储先验概率和条件概率
    """

    def __init__(self, x, y):
        #15
        self.N = len(x)  # 样本数 —— 先验概率的分母
        #2
        self.n = len(x[0])  # 维度数

        # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
        self.y_list = list(set(y)) # [-1, 1]
        # {-1: 0, 1, 1}
        self.y_mapping = {c: i for i, c in enumerate(self.y_list)} 
        #self.x_list = [
        #[1, 2, 3],      # 特征 j=0 的所有唯一取值
        #['S', 'M', 'L'] # 特征 j=1 的所有唯一取值
        #]
        self.x_list = [list(set(x[i][j] for i in range(self.N))) for j in range(self.n)]
        #[S, M, L] -> {S:0, M:1, L:2}
        """
        self.x_mapping = [
    {1: 0, 2: 1, 3: 2},      # 特征 0 的值映射
    {'S': 0, 'M': 1, 'L': 2} # 特征 1 的值映射
]

        """
        self.x_mapping = [{c: i for i, c in enumerate(self.x_list[j])} for j in range(self.n)]

        # 计算可能取值数
        #2
        self.K = len(self.y_list)  # Y的可能取值数 
        # [3,3]
        self.Sj = [len(self.x_list[j]) for j in range(self.n)]  # X各个特征的可能取值数

        # 计算：P(Y=ck) —— 先验概率的分子、条件概率的分母
        #[0, 0]
        table1 = [0] * self.K
        # y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
        #15 -> {-1:0, 1:1} ->
        for i in range(self.N):
            table1[self.y_mapping[y[i]]] += 1
        # [6, 9]

        # 计算：P(Xj=ajl|Y=ck) —— 条件概率的分子
        # n=2, k= 2, Sj=[3,3] , N = 15
        table2 = [[[0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for i in range(self.N):
            for j in range(self.n):
                table2[j][self.y_mapping[y[i]]][self.x_mapping[j][x[i][j]]] += 1
        #  [
        # [[0, 0, 0], [0, 0, 0]],
        #  [[0, 0, 0], [0, 0, 0]]  ]
        """
        table2 = [
    # 特征 0 (X₁)
    [[3, 2, 1],   # Y=-1: X₁=1出现3次, X₁=2出现2次, X₁=3出现1次
     [2, 3, 4]],  # Y=1:  X₁=1出现2次, X₁=2出现3次, X₁=3出现4次
    
    # 特征 1 (X₂)
    [[3, 2, 1],   # Y=-1: X₂='S'出现3次, 'M'出现2次, 'L'出现1次
     [1, 4, 4]]   # Y=1:  X₂='S'出现1次, 'M'出现4次, 'L'出现4次
]

        """

        # 计算先验概率
        self.prior = [0.0] * self.K
        for k in range(self.K):
            self.prior[k] = table1[k] / self.N

        # 计算条件概率
        self.conditional = [[[0.0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for j in range(self.n):
            for k in range(self.K):
                for t in range(self.Sj[j]):
                    self.conditional[j][k][t] = table2[j][k][t] / table1[k]


    def predict(self, x):
        best_y, best_score = 0, 0
        for k in range(self.K):
            score = self.prior[k]
            for j in range(self.n):
                if x[j] in self.x_mapping[j]:
                    score *= self.conditional[j][k][self.x_mapping[j][x[j]]]
                else:
                    score *= 0
            if score > best_score:
                best_y, best_score = self.y_list[k], score
        return best_y


class NaiveBayesAlgorithmHashmap:
    """朴素贝叶斯算法（仅支持离散型数据）

    使用哈希表存储先验概率和条件概率
    """

    def __init__(self, x, y):
        self.N = len(x)  # 样本数

        self.n = len(x[0])  # 维度数

        count1 = collections.Counter(y)  # 先验概率的分子，条件概率的分母
        count2 = [collections.Counter() for _ in range(self.n)]  # 条件概率的分子
        for i in range(self.N):
            for j in range(self.n):
                count2[j][(x[i][j], y[i])] += 1

        # 计算先验概率和条件概率
        self.prior = {k: v / self.N for k, v in count1.items()}
        self.conditional = [{k: v / count1[k[1]] for k, v in count2[j].items()} for j in range(self.n)]

    def predict(self, x):
        best_y, best_score = 0, 0
        for y in self.prior:
            score = self.prior[y]
            for j in range(self.n):
                score *= self.conditional[j][(x[j], y)]
            if score > best_score:
                best_y, best_score = y, score
        return best_y


if __name__ == "__main__":
    dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
                (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
                (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
               [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
    naive_bayes_1 = NaiveBayesAlgorithmHashmap(*dataset)
    print(naive_bayes_1.predict([2, "S"]))

    naive_bayes_2 = NaiveBayesAlgorithmArray(*dataset)
    print(naive_bayes_2.predict([2, "S"]))




