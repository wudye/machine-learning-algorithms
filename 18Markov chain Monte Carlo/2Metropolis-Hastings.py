import numpy as np

"""
“接受-拒绝采样”虽然好用，但它有一个致命弱点：如果目标分布在高维空间，很难找到一个合适的“盖子”分布（gx）和常数 c，导致接受率极低。
核心思想：不用“盖子”，改用“散步”
接受-拒绝采样：像是在整个盒子里随机撒点，不合适的就扔掉。
M-H 算法：像是在山山上散步。你当前在位置 xi，通过一个简单的随机规则（建议分布）提议去下一个位置 x'。
如果新位置“更好”（概率密度更高），你就大概率跳过去；如果“更差”，你也有一定概率留原地。
初始化：随机选一个起点 x0。
提议（Propose）：从一个简单的分布（通常是以当前位置为中心的正态分布）中生成一个候选点x' 。
计算接受率 a
判定（Accept/Reject）：
    生成随机数 u-U(0-1)。
    如果 u<=a，接受新点：xi+1=x'。
    关键点：如果拒绝，原地不动：xi+1 = xi（样本里增加一个重复的旧点）。
重复：迭代成千上万次，最后留下的路径点就是样本。
"""
def metropolis_hastings_method(d1, func, m, n, x0, random_state=0):
    """Metroplis-Hastings算法抽取样本

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

    # 循环执行n次迭代
    for k in range(n):
        # 按照建议分布q(x,x')随机抽取一个候选状态
        # q(x,x')为均值为x，方差为1的正态分布
        """
        它以当前位置 x0 为中心（均值）。
        加上一个标准正态分布的随机扰动。
        这意味着：你向左走1步和向右走1步的概率是完全对称
        a = min(1, f(x1) / f(x0) * q(x0, x1) / q(x1, x0)) 这里的 q(x0, x1) 和 q(x1, x0) 是对称的，所以它们会抵消掉，最终接受率 a 就简化为 min(1, f(x1) / f(x0))。
         这也是为什么在 M-H 算法中，建议分布通常选择对称的分布（如正态分布）的原因之一：它简化了接受率的计算。
        """
        x1 = np.random.multivariate_normal(x0, np.diag([1] * n_features), 1)[0]

        # 计算接受概率
        a = min(1, d1(x1) / d1(x0))

        # 从区间(0,1)中按均匀分布随机抽取一个数u
        u = np.random.rand()

        # 若u<=a，则转移状态；否则不转移
        if u <= a:
            x0 = x1

        # 收集样本集合
        """
        m (收敛步数 / Burn-in)：
        算法最开始的样本可能受初始值 x0 影响很大，还没进入高概率的核心区域。
        if k >= m: 这一行代码会把前 m 个样本丢弃，只记录马尔可夫链进入平稳状态后的样本。
        马尔可夫链最终会达到一个平稳分布（Stationary Distribution），这个分布就是我们的目标分布。但是，算法刚开始时的起点 x0 是我们随机选的：
        如果 x0 选在了一个概率极低的“荒漠”地带，算法需要走很多步才能“爬”到高概率的“山峰”区域。
        在“爬山”过程中产生的样本并不符合目标分布的统计特性。
        丢弃前 m 步，是为了让链有足够的时间“忘记”那个随意的起点，进入真正的目标区域。
        确保样本的“代表性”
        M-H 算法生成的样本序列是一条连贯的路径。
        预热期（Burn-in）：就像是运动员比赛前的热身。热身时的状态不计入正式比赛成绩。
        如果不丢弃 m，你的样本统计结果（均值、方差）会被起始点偏离中心的过程拉低或带偏，导致结果不准。
        """
        if k >= m:
            samples.append(x0)
            sum_ += func(x0)

    return samples, sum_ / (n - m)


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt


    """
    f(x1,x2) = x1 * e^(-x2)  if 0 < x1 < x2 else 0
    在坐标系上，这代表一个由直线 x1 = 0（y轴）和 x1 = x2 围成的无穷大的三角形区域。
    相关性：这两个变量是不独立的。
    因为 x1的取值范围直接依赖于 x2的当前值（x1不能超过 x2）。
    采样难度：由于存在 x1 < x2 的限制，使用简单的“独立抽样”会非常低效（很多点会落在三角形外被拒绝）。这正是为什么需要
     Metropolis-Hastings 或 Gibbs Sampling 的原因。
    """
    def d1_pdf(x):
        """随机变量x=(x_1,x_2)的联合概率密度"""
        return x[0] * pow(np.e, -x[1]) if 0 < x[0] < x[1] else 0


    def f(x):
        """目标求均值函数"""
        return x[0] + x[1]


    samples, avg = metropolis_hastings_method(d1_pdf, f, m=1000, n=11000, x0=[5, 8])

    #print(samples)  # [array([0.39102823, 0.58105655]), array([0.39102823, 0.58105655]), ...]
    #print("样本目标函数均值:", avg)  # 4.720997790412456


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