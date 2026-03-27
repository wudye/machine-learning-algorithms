
import collections

class NaiveBayesSimple:
    """朴素贝叶斯 - 简化版（使用字典）"""

    def __init__(self, x, y):
        self.N = len(y)

        # 统计类别计数：{类别: 次数}
        self.class_count = collections.Counter(y)

        # 统计特征条件计数：{特征索引: {(特征值, 类别): 次数}}
        self.feature_count = [collections.Counter() for _ in range(len(x[0]))]

        for i in range(self.N):
            for j in range(len(x[0])):
                self.feature_count[j][(x[i][j], y[i])] += 1
        print(self.feature_count)

    def predict(self, sample):
        best_class, best_score = None, 0

        for y_val, y_count in self.class_count.items():
            # 先验概率 P(Y)
            score = y_count / self.N

            # 条件概率 P(Xj|Y)
            for j, feature_val in enumerate(sample):
                count = self.feature_count[j][(feature_val, y_val)]
                score *= count / y_count
                print(score)

            if score > best_score:
                best_class, best_score = y_val, score

        return best_class



if __name__ == "__main__":
    dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
                (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
                (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
               [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
    naive_bayes_1 = NaiveBayesSimple(*dataset)
    print(naive_bayes_1.predict([2, "S"]))

