import numpy as np

"""
它不是一次性跳到一个全新的多维坐标点，而是每次只改变坐标的一个维度（分量），轮流更新。
"""
def single_component_metropolis_hastings_method(d1, func, m, n, x0, random_state=0):
    """单分量Metroplis-Hastings算法抽取样本

    :param d1: 目标概率分布的概率密度函数
    :param func: 目标求均值函数
    :param x0: 初值（定义域中的任意一点即可）
    :param m: 收敛步数
    :param n: 迭代步数
    :param random_state: 随机种子
    :return: 随机样本列表,随机样本的目标函数均值
    """
    np.random.seed(random_state)

    samples = []  # 随机样本列表
    sum_ = 0  # 目标求均值函数的和

    n_features = len(x0)
    j = 0  # 当前正在更新的分量

    # 循环执行n次迭代
    for k in range(n):
        # 按照建议分布q(x,x')随机抽取一个候选状态
        # q(x,x')为均值为x，方差为1的正态分布
        x1 = x0.copy()
        """
        一维正态分布采样
        均值 为 x0[j]，方差为 1 的正态分布。
         这意味着：你向左走1步和向右走1步的概率是完全对称的。
         这里的 x0[j] 是当前状态在第 j 个维度上的值。每次迭代，我们只尝试改变这个维度的值，而其他维度保持不变。
         这样做的好处是：对于高维分布，一次性改变所有维度很难被“接受”（容易跳到概率极低的区域）。每次只改一个维度，接受率通常会更高，路径更稳健。
        """
        x1[j] = np.random.multivariate_normal([x0[j]], np.diag([1]), 1)[0][0]

        # 计算接受概率
        a = min(1, d1(x1) / d1(x0))

        # 从区间(0,1)中按均匀分布随机抽取一个数u
        u = np.random.rand()

        # 若u<=a，则转移状态；否则不转移
        if u <= a:
            x0 = x1

        # 收集样本集合
        if k >= m:
            samples.append(x0)
            sum_ += func(x0)

        """
        它维护一个指针 j。
        第一步：只尝试改变x1，保持x2,x3...不变。
        第二步：只尝试改变 x2，保持x1,x3..不变。
        如此循环往复。
        好处：对于高维分布，一次性改变所有维度很难被“接受”（容易跳到概率极低的区域）。每次只改一个维度，接受率通常会更高，路径更稳健。
        """
        j = (j + 1) % n_features

    return samples, sum_ / (n - m)


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt


    def d1_pdf(x):
        """随机变量x=(x_1,x_2)的联合概率密度"""
        return x[0] * pow(np.e, -x[1]) if 0 < x[0] < x[1] else 0


    def f(x):
        """目标求均值函数"""
        return x[0] + x[1]


    samples, avg = single_component_metropolis_hastings_method(d1_pdf, f, m=1000, n=11000, x0=[5, 8])

    print(samples)  # [[0.6497854877644121, 1.597507333170185], [0.6497854877644121, 1.597507333170185], ...]
    print("样本目标函数均值:", avg)  # 4.7348085753536076


    def draw_distribution():
        """绘制总体概率密度函数的图（优化版）"""
        x = np.arange(0, 10, 0.1)
        y = np.arange(0, 10, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # 生成目标概率密度曲面矩阵 Z
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = d1_pdf([X[i, j], Y[i, j]])
        print(Z)

        # 1. 2D热力平面图
        fig1 = plt.figure(figsize=(7, 5))
        # origin='lower' 将 (0,0) 移到左下角；extent 将横纵轴标度对应到 0~10 的实际值
        plt.imshow(Z, cmap="viridis", origin="lower", extent=[0, 10, 0, 10])
        plt.colorbar(label='Probability Density')
        plt.title("Target Distribution (2D Heatmap)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

        # 2. 3D曲面图
        fig2 = plt.figure(figsize=(9, 6))
        ax = fig2.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap="viridis",
                               edgecolor='none', alpha=0.9)
        fig2.colorbar(surf, shrink=0.5, aspect=10, label='Density')
        ax.set_title("Target Distribution (3D Surface)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("Density Z")
        # 调整观察初始角度
        ax.view_init(elev=30, azim=-45)
        plt.show()


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


    draw_distribution()
    draw_sample(samples)