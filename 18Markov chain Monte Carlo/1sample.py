from abc import ABC
from abc import abstractmethod

import numpy as np


class BaseDistribution(ABC):
    """随机变量分布的抽象基类"""

    @abstractmethod
    def pdf(self, x):
        """计算概率密度函数"""
        pass

    def cdf(self, x):
        """计算分布函数"""
        raise ValueError("未定义分布函数")


class UniformDistribution(BaseDistribution):
    """均匀分布

    :param a: 左侧边界
    :param b: 右侧边界
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, x):
        if self.a < x < self.b:
            return 1 / (self.b - self.a)
        else:
            return 0

    def cdf(self, x):
        if x < self.a:
            return 0
        elif self.a <= x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return 1


class GaussianDistribution(BaseDistribution):
    """高斯分布（正态分布）

    :param u: 均值
    :param s: 标准差
    """

    def __init__(self, u, s):
        self.u = u
        self.s = s

    def pdf(self, x):
        return pow(np.e, -1 * (pow(x - self.u, 2)) / 2 * pow(self.s, 2)) / (np.sqrt(2 * np.pi * pow(self.s, 2)))

# Inverse Transform Sampling
def direct_sampling_method(distribution, n_samples, a=-1e5, b=1e5, tol=1e-6, random_state=0):
    """直接抽样法抽取样本

    通常我们是知道一个数值 x，去查它的累积概率 p(X<=x)。而这个算法是反向操作：先生成一个0到1之间的随机数y(这代表累积概率）。
    利用分布函数F(x)的单调性，寻找哪一个x对应的累积概率刚好等于y。
    这个解出来的
    就是我们要的样本。
    :param distribution: 定义分布函数的概率分布
    :param n_samples: 样本数
    :param a: 定义域左侧边界
    :param b: 定义域右侧边界
    :param tol: 容差
    :param random_state: 随机种子
    :return: 随机样本列表
    """
    np.random.seed(random_state)

    samples = []
    for _ in range(n_samples):
        y = np.random.rand()

        # 二分查找解方程：F(x) = y
        l, r = a, b
        while r - l > tol:
            m = (l + r) / 2
            # 如果当前的累积概率太大，说明 x 应该在左半部分，于是缩小右边界
            if distribution.cdf(m) > y:
                r = m
            else:
                l = m

        samples.append((l + r) / 2)

    return samples


def accept_reject_sampling_method(d1, d2, c, n_samples, a=-1e5, b=1e5, tol=1e-6, random_state=0):
    """接受-拒绝抽样法
    接受-拒绝采样法 (Accept-Reject Sampling)核心思想是：当你无法直接从复杂的目标分布 (d1) 中抽样时，先从一个简单的建议分布 (d2) 中抽样，
    然后根据一定的概率决定是“接受”还是“拒绝”这个样本。
    核心数学原理是：如果存在一个常数 c，使得对于所有 x 都满足 f(x) <= c * g(x)，其中 f(x) 是目标分布的概率密度函数，g(x) 是建议分布的概率密度函数，那么我们就可以使用接受-拒绝采样法。
     这个不等式的意义是：建议分布 g(x) 的概率密度函数必须足够“高”，以覆盖目标分布 f(x) 的所有可能取值。常数 c 的作用是将建议分布“拉高”，形成一个信封（Envelope），把目标分布完全盖在下面。
     采样步骤如下：
    1. 从建议分布 g(x) 中抽取一个样本 x。
    2. 生成一个均匀分布的随机数 u，范围在 0 和 1 之间。
    3. 如果 u < f(x) / (      c * g(x) )，则接受这个样本 x；否则拒绝它。
     这个接受概率的计算方式确保了最终接受的样本分布与目标分布 f(x) 一致。因为接受的概率是根据目标分布和建议分布的比例来决定的，
     所以最终接受的样本会更倾向于目标分布的高概率区域。
    现实情况：很多时候我们根本写不出 d1的累积分布函数（CDF）或者它的逆函数，但我们能轻易写出它的概率密度函数（PDF）。
    解决方案：找一个 CDF 很简单的分布（如均匀分布或指数分布）作为 d2，用逆变换法抽d2，再通过“接受-拒绝”逻辑把它变成d1的样本。

    :param d1: 目标概率分布
    :param d2: 建议概率分布
    :param c: 参数c
    :param n_samples: 样本数
    :param a: 建议概率分布定义域左侧边界
    :param b: 建议概率分布定义域右侧边界
    :param tol: 容差
    :param random_state: 随机种子
    :return: 随机样本列表
    """
    np.random.seed(random_state)

    samples = []
    """
    根据算法原理，样本被接受的概率是：
    P(accept) = 1 / c 
    如果你之前计算的 c，那么接受率大约只有 41.8%。
    这意味着如果你只抽 n_samples 个建议样本，最后可能只有不到一半能用。
    这意味着你每生成 100个 建议样本，平均只有约 42个 能通过测试存入 samples
    c 越大，“盖子”就离目标曲线越远，浪费的间隙就越多，采样效率就越低。
    
    """
    waiting = direct_sampling_method(d2, n_samples * 2, a=a, b=b, tol=tol, random_state=random_state)  # 直接抽样法得到建议分布的样本
    while len(samples) < n_samples:
        if not waiting:
            """
            (n_samples - len(samples))：计算缺口。即：你总共想要多少个，减去目前已经成功拿到（接受）了多少个。
            假设你需要 100 个样本 (n_samples=100)。
            你在进入循环前先抽了 200 个候选者。
            在 while 循环处理这 200 个人的过程中，可能只有 30 个人被接受了。
            此时 waiting 空了，但你手里只有 30 个样，还差 70 个。
            再次去抽 140个候选者来继续尝试。
            """
            waiting = direct_sampling_method(d2, (n_samples - len(samples)) * 2, a, b)

        x = waiting.pop()
        u = np.random.rand()
        if u <= (d1.pdf(x) / (c * d2.pdf(x))):
            samples.append(x)

    return samples


if __name__ == "__main__":
    d1 = GaussianDistribution(0, 1)
    d2 = UniformDistribution(-3, 3)

    """
    将高度为 1/6 的水平线（均匀分布）拉升到足以覆盖标准正态分布峰值所需的倍数
    f(x) <= c * g(x)  =>  f(x) / (c * g(x) )
    f(x)  是你想采样的目标分布（如标准正态分布）。
    g(x)  是你选择的建议分布（如均匀分布），它应该在目标分布的定义域内，并且能够覆盖目标分布的所有可能取值。
    是你容易抽样的建议分布（如均匀分布）。
    c 的作用：它是为了把 gx“拉高”，形成一个信封（Envelope），把 fx完全盖在下面。
    if u < f(x) / (c * g(x)):    # 接受这个样本
    """
    c = (1 / np.sqrt(2 * np.pi)) / (1 / 6)  # 计算c的最小值

    print(c)
    samples = accept_reject_sampling_method(d1, d2, c, 10, a=-3, b=3)
    print([round(v, 2) for v in samples])  # [0.17, -0.7, -0.37, 0.88, -0.46, 0.27, 0.62, 0.29, 0.27, 0.62]