from copy import copy

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

class AdaBoostRegressor:

    def __init__(self, X, Y, weak_clf, M=10):
        self.X, self.Y = X, Y
        self.weak_clf = weak_clf
        self.M = M

        self.n_samples = len(self.X)

        self.G_list = []

        self._train()

    def _train(self):
        r =[self.Y[i] for i in range(self.n_samples)]  # 初始化残差列表

        for m in range(self.M):
            self.weak_clf.fit(self.X, r)
            self.G_list.append(copy(self.weak_clf))

            predict = self.weak_clf.predict(self.X)
            for i in range(self.n_samples):
                r[i] -= predict[i]  # 更新残差列表

    def predict(self, x):
        return sum(self.G_list[i].predict([x])[0] for i in range(len(self.G_list)))

if __name__ == "__main__":
    # ---------- 《统计学习方法》例8.2 ----------
    dataset = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
               [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]]

    seg = AdaBoostRegressor(dataset[0], dataset[1], DecisionTreeRegressor(max_depth=1), M=6)
    r = sum((seg.predict(dataset[0][i]) - dataset[1][i]) ** 2 for i in range(10))
    print("MSE:", r)

    # ---------- sklearn波士顿房价 ----------
    import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    Y = raw_df.values[1::2, 2]

    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    seg = AdaBoostRegressor(x1, y1, DecisionTreeRegressor(max_depth=1), M=50)
    r = sum((seg.predict(x2[i]) - y2[i]) ** 2 for i in range(len(x2)))
    print("MSE:", r)


    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    Y = raw_df.values[1::2, 2]

    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)
    seg = GradientBoostingRegressor(n_estimators=50, learning_rate=1, max_depth=1, random_state=0, loss='squared_error')
    seg.fit(x1, y1)
    r = sum((seg.predict([x2[i]])[0] - y2[i]) ** 2 for i in range(len(x2)))
    print("MSE:", r)