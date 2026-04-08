import math
from collections import Counter

from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.model_selection import train_test_split


class GaussianMixture:

    def __init__(self, X, init_mean, n_components, max_iter=100):
        self.X = X
        self.n_components = n_components
        self.max_iter = max_iter

        self.n_samples = len(X)

        self.alpha = [1 / self.n_components] * self.n_components  # 分模型权重
        self.means = init_mean  # 均值
        self.sigma = [1.0] * self.n_components  # 方差
        self._train()

    def _train(self):
        for _ in range(self.max_iter):
            # E 步回答"样本属于哪个高斯", gamma 的值取决于当前参数
            gamma = [[0] * self.n_components for _ in range(self.n_samples)]
            for j in range(self.n_samples):
                sum_ = 0
                for k in range(self.n_components):
                    gamma[j][k] = self.alpha[k] * self._count_gaussian(self.X[j], k)
                    sum_ += gamma[j][k]
                for k in range(self.n_components):
                    gamma[j][k] /= sum_

            # M 步回答"那高斯应该长什么样"
            means_new = [0.0] * self.n_components
            for k in range(self.n_components):
                sum1 = 0  # 分子
                sum2 = 0  # 分母
                for j in range(self.n_samples):
                    sum1 += gamma[j][k] * self.X[j]
                    sum2 += gamma[j][k]
                means_new[k] = sum1 / sum2

            sigma_new = [1.0] * self.n_components
            for k in range(self.n_components):
                sum1 = 0  # 分子
                sum2 = 0  # 分母
                for j in range(self.n_samples):
                    sum1 += gamma[j][k] * math.pow(self.X[j] - self.means[k], 2)
                    sum2 += gamma[j][k]
                sigma_new[k] = sum1 / sum2

            alpha_new = [1 / self.n_components] * self.n_components
            for k in range(self.n_components):
                sum1 = 0  # 分子
                for j in range(self.n_samples):
                    sum1 += gamma[j][k]
                alpha_new[k] = sum1 / self.n_samples
            """
            # 对数似然收敛判断
            ll = self._log_likelihood()
            if abs(ll - self._prev_ll) < 1e-6:  # 真正的停止条件
                print(f"收敛于第 {m+1} 轮")
                break
            self._prev_ll = ll
            """
            self.alpha = alpha_new
            self.means = means_new
            self.sigma = sigma_new

    def _count_gaussian(self, x, k):
        """计算高斯密度函数"""
        return math.pow(math.e, -math.pow(x - self.means[k], 2) / (2 * self.sigma[k])) / (
            math.sqrt(2 * math.pi * self.sigma[k]))

    def predict(self, x):
        best_k, best_g = -1, 0
        for k in range(self.n_components):
            g = self.alpha[k] * self._count_gaussian(x, k)
            if g > best_g:
                best_k, best_g = k, g
        return best_k

    # 从推导角度 EM 应该用对数似然收敛来停止
    def _log_likelihood(self):
        """计算当前参数下的对数似然"""
        ll = 0
        for j in range(self.n_samples):
            # P(x_j) = Σ α_k · N(x_j | μ_k, σ²_k)
            p_x = sum(
                self.alpha[k] * self._count_gaussian(self.X[j], k)
                for k in range(self.n_components)
            )
            ll += math.log(p_x)
        return ll



if __name__ == "__main__":

    X, Y = make_blobs(n_samples=1500, n_features=1,
                      centers=[[-2], [2]], cluster_std=1, random_state=0)

    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)
    x1 = [float(elem[0]) for elem in x1]
    x2 = [float(elem[0]) for elem in x2]

    n_components = 2  # 类别数
    n_samples = len(x1)  # 样本数
    n_samples_of_type = Counter(y1)  # 每个类别的样本数
    print(n_samples_of_type)

    # 计算各个类别的平均值
    means = [[0] for _ in range(n_components)]
    for yi in range(n_components):
        means[yi][0] = sum(x1[i] for i in range(n_samples) if y1[i] == yi) / n_samples_of_type[yi]
    print(means)
    # 训练高斯混合模型

    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit([[x] for x in x1])
    init_mean = [kmeans.cluster_centers_[i][0] for i in range(n_components)]
    """
    gmm = GaussianMixture(x1, [0.1, 0.2], n_components)

    # 计算高斯混合模型的每个类别对应的实际类别
    """
    真实类别:   类别0 (中心=-2)    类别1 (中心=2)

    EM 不知道编号规则:
      可能 高斯0 → 对应类别0    高斯1 → 对应类别1    ← 刚好对上
      也可能 高斯0 → 对应类别1    高斯1 → 对应类别0    ← 对不上！
    means = [[-2.0], [2.0]]   # means[0]=-2.0 (类别0), means[1]=2.0 (类别1)
    gmm.means = [2.1, -1.9]   # 高斯0的中心=2.1, 高斯1的中心=-1.9
    [[m] for m in gmm.means] gmm.means = [2.1, -1.9]  →  [[2.1], [-1.9]] 变成二维列表，匹配 means 的格式
    pairwise_distances_argmin 对  means 中的每个元素，在第二个参数中找最近的元素，返回索引
    结果：[1, 0]
    """
    mapping = {t1: t2 for t1, t2 in enumerate(pairwise_distances_argmin(means, [[m] for m in gmm.means]))}
    print(mapping)

    # 计算准确率
    correct = 0
    for x, actual_y in zip(x2, y2):
        predict_y = mapping[gmm.predict(x)]
        if predict_y == actual_y:
            correct += 1
    print("准确率:", correct / len(x2))