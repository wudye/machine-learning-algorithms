import numpy as np

"""
吉布斯抽样（Gibbs Sampling）核心逻辑是：如果你很难直接抽取 (x1,x2)，但你知道在给定 
 x2时x1的分布（条件分布），以及在给定 x1时x2
 的分布（条件分布），那么就可以通过交替抽取来实现。
"""
def gibbs_sampling_method(mean, cov, func, n_samples, m=1000, random_state=0):
    """吉布斯抽样算法在二元正态分布中抽取样本

    与np.random.multivariate_normal方法类似

    :param mean: n元正态分布的均值
    :param cov: n元正态分布的协方差矩阵
    :param func: 目标求均值函数
    :param n_samples: 样本量
    :param m: 收敛步数
    :param random_state: 随机种子
    :return: 随机样本列表
    """
    np.random.seed(random_state)

    # 选取初始样本
    x0 = mean

    samples = []  # 随机样本列表
    sum_ = 0  # 目标求均值函数的和

    # 循环执行n次迭代
    for k in range(m + n_samples):
        # 根据满条件分布逐个抽取样本
        """
        吉布斯采样的关键在于利用其条件分布（二元正态的条件分布依然是正态分布）：
        假设我们要采样的目标分布是二元正态分布：
        (x1, x2) ~ N(μ, Σ)，其中 μ 是均值向量，Σ 是协方差矩阵。
        对于二元正态分布 N(μ, Σ)，如果我们固定其中一个变量（比如 x2），另一个变量（x1）的条件分布仍然是正态分布，且其均值和方差可以通过协方差矩阵 Σ 的元素计算出来。
        具体来说：
        - x1 的条件分布 P(x1 | x2) 是一个均值为 μ1 + (Σ12 / Σ22) * (x2 - μ2)，方差为 Σ11 - (Σ12^2 / Σ22) 的正态分布。
        - x2 的条件分布 P(x2 | x1) 是一个均值为 μ2 + (Σ12 / Σ11) * (x1 - μ1)，方差为 Σ22 - (Σ12^2 / Σ11) 的正态分布。
        因此，在每次迭代中，我们先根据当前的 x2 来抽取一个新的 x1，然后再根据新的 x1 来抽取一个新的 x2。通过不断交替更新这两个变量，我们的样本序列会逐渐收敛到目标分布 N(μ, Σ)。
        """
        x0[0] = np.random.multivariate_normal([x0[1] * cov[0][1]], np.diag([1 - pow(cov[0][1], 2)]), 1)[0][0]
        x0[1] = np.random.multivariate_normal([x0[0] * cov[0][1]], np.diag([1 - pow(cov[0][1], 2)]), 1)[0][0]

        # 收集样本集合
        if k >= m:
            samples.append(x0.copy())
            sum_ += func(x0)

    return samples, sum_ / n_samples


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt


    def f(x):
        """目标求均值函数"""
        return x[0] + x[1]


    samples, avg = gibbs_sampling_method([0, 0], [[1, 0.5], [0.5, 1]], f, n_samples=10000)

    print(samples)  # [[-2.0422584903207794, -2.5037388977869997], [-1.211915315832784, -1.4359343041313015], ...]
    print("样本目标函数均值:", avg)  # 0.0016714992469064399



    def draw_sample(samples):
        """绘制样本概率密度分布的图（优化版）"""
        samples_arr = np.array(samples)
        x_samples = samples_arr[:, 0]
        y_samples = samples_arr[:, 1]

        # 直接借助 numpy 计算二维直方图，比手动两层 for 循环统计快得多
        bins = np.arange(0, 10.1, 0.1)
        H, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=(bins, bins))

        # np.histogram2d返回的矩阵需要转置才能与imshow的横纵坐标对应
        H = H.T

        # 1. 2D热力平面图
        fig1 = plt.figure(figsize=(7, 5))
        plt.imshow(H, cmap="viridis", origin="lower", extent=[0, 10, 0, 10], interpolation='nearest')
        plt.colorbar(label='Sample Count')
        plt.title("MCMC Samples Distribution (2D Heatmap)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

        # 2. 3D曲面/网格图
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        fig2 = plt.figure(figsize=(9, 6))
        ax = fig2.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, H, rstride=2, cstride=2, cmap="viridis",
                               edgecolor='none', alpha=0.9)
        fig2.colorbar(surf, shrink=0.5, aspect=10, label='Count')
        ax.set_title("MCMC Samples Density (3D Surface)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("Sample Count")
        ax.view_init(elev=30, azim=-45)
        plt.show()


    draw_sample(samples)