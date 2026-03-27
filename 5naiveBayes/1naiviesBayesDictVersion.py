from collections import Counter

class NaiveBayesDict:
    def __init__(self, x, y):
        self.N = len(y)
        self.n = len(x[0])

        self.class_count = Counter(y)
        self.feature_count = [Counter() for _ in range(self.n)]

        for i in range(self.N):
            for j in range(self.n):
                self.feature_count[j][(x[i][j], y[i])] += 1

    def predict(self, sample):
        best_score, best_class = 0, None
        for y_val, y_count in self.class_count.items():
            score = y_count / self.N
            for j, feature_val in enumerate(sample):
                count = self.feature_count[j][(feature_val, y_val)]
                score *= count / y_count
            if score > best_score:
                best_score, best_class = score, y_val

        return best_class


if __name__ == "__main__":
    dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
                (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
                (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
               [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
    naive_bayes_1 = NaiveBayesDict(*dataset)
    print(naive_bayes_1.predict([2, "S"]))
