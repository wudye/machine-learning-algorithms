import math
from copy import copy

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost:

    def  __init__(self, X, Y, weak_clf, M=10):
        self.X, self.Y = X, Y
        self.weak_clf = weak_clf
        self.M = M

        self.n_samples = len(self.X)
        self.n_features = len(self.X[0])

        self.G_list = []
        self.a_list = []

        self._train()

    """
        推导中的数学符号                    →    代码实现
    ─────────────────────────────────────────────────
    G_m(x)  [弱分类器]                 →    self.weak_clf.fit(..., sample_weight=D)
                                            self.weak_clf.predict(self.X)
    
    α_m     [分类器系数/权重]           →    a = 0.5 * math.log((1-error)/error)
    
    e_m     [加权误差率]               →    error = sum(D[i] for i ...)
    
    w_i     [样本权重]                 →    D (列表)
    
    F(x)    [最终强分类器]              →    sign(Σ a_m * G_m(x)) 

    """
    def _train(self):
        # D is the weights of the samples, initialized to 1/n_samples
        D = [1 / self.n_samples] * self.n_samples

        # fx is the weighted sum of the weak classifiers' predictions, initialized to 0
        fx = [0] * self.n_samples

        for m in range(self.M):

            # G_m(x)  [弱分类器]
            # 决策树等价于手动穷举, max_depth=1 的决策树 = 决策树桩 = 穷举所有阈值
            self.weak_clf.fit(self.X, self.Y, sample_weight=D)

            predict = self.weak_clf.predict(self.X)

            """
            error = 0
            for i in range(self.n_samples):
                if np.sign(predict[i]) != self.Y[i]:   # 如果第i个样本被错分
                    error += D[i]           
            """
            error = sum(D[i] for i in range(self.n_samples) if np.sign(predict[i]) != self.Y[i])
            a = 0.5 * math.log((1 - error) / error)

            """
            copy shallow copy
            fit() 之后，决策树内部状态主要存储在这些属性中：
            DecisionTreeClassifier
            ├── tree_          ← 核心数据：树结构（sklearn 的 C 扩展对象）
            ├── n_features_in_
            ├── n_classes_
            ├── classes_
            └── ...
            下一轮 fit() 调用时，sklearn 的决策树会重建整个 tree_ 对象，而不是修改原有 tree_ 的内部属性。
            """
            self.G_list.append(copy(self.weak_clf))
            self.a_list.append(a)

            """
            D_new = []
            for i in range(self.n_samples):
                exponent = -a * self.Y[i] * predict[i]
                D_new.append(D[i] * math.e ** exponent)
            D = D_new
            """
            D = [D[i] * pow(math.e, -a * self.Y[i] * predict[i]) for i in range(self.n_samples)]
            Z = sum(D)
            D = [v / Z for v in D]

            wrong_num = 0
            for i in range(self.n_samples):
                fx[i] += a * predict[i]
                if np.sign(fx[i]) != self.Y[i]:
                    wrong_num += 1
            print(f"Round {m + 1}: error = {error:.4f}, wrong_num = {wrong_num}")
            if wrong_num == 0:
                break

    """
    F(x) = sign(α₁·G₁(x) + α₂·G₂(x) + ... + αₘ·Gₘ(x))
               ↑           ↑                ↑
            copy_v1     copy_v2          copy_vm
           （第1轮模型） （第2轮模型）    （第m轮模型）
    def predict(self, x):
        total = 0
        for i in range(len(self.G_list)):
            weak_pred = self.G_list[i].predict([x])   # 第i个弱分类器的预测
            weighted  = self.a_list[i] * weak_pred    # 乘以系数α
            total += weighted                          # 累加
        return np.sign(total)                          # 取符号
    """
    def predict(self, x):
        return np.sign(sum(self.a_list[i] * self.G_list[i].predict([x]) for i in range(len(self.G_list))))


if __name__ == "__main__":
    dataset = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
               [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]]

    clf = AdaBoost(dataset[0], dataset[1], DecisionTreeClassifier(max_depth=1))
    correct = 0
    for ii in range(10):
        if clf.predict([ii]) == dataset[1][ii]:
            correct += 1
    print(f"Accuracy: {correct / 10:.4f}")

    X, Y = load_breast_cancer(return_X_y=True)
    Y = np.where(Y == 0, -1, 1)  # 将标签转换为 -1 和 1
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = AdaBoost(x1, y1, DecisionTreeClassifier(max_depth=1))
    correct = 0
    for i in range(len(x2)):
        if clf.predict(x2[i]) == y2[i]:
            correct += 1
    print("accuracy:", correct / len(x2))

    X, Y = load_breast_cancer(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=10,random_state=0)
    clf.fit(x1, y1)
    print("Accuracy:", clf.score(x2, y2))  # 预测正确率: 0.9736842105263158
