import random
import math


def forward(a, b, pi, obs):
    """前向算法, 返回所有时刻的 alpha 和 P(O|λ)"""
    T = len(obs)
    N = len(a)
    alpha = [[0] * N for _ in range(T)]

    # 初始化
    for i in range(N):
        alpha[0][i] = pi[i] * b[i][obs[0]]

    # 递推
    for t in range(1, T):
        for i in range(N):
            alpha[t][i] = sum(alpha[t - 1][j] * a[j][i] for j in range(N)) * b[i][obs[t]]

    P = sum(alpha[T - 1])
    return alpha, P


def backward(a, b, pi, obs):
    """后向算法, 返回所有时刻的 beta 和 P(O|λ)"""
    T = len(obs)
    N = len(a)
    beta = [[0] * N for _ in range(T)]

    # 初始化
    for i in range(N):
        beta[T - 1][i] = 1.0

    # 递推 (从后往前)
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t][i] = sum(a[i][j] * b[j][obs[t + 1]] * beta[t + 1][j] for j in range(N))

    P = sum(pi[i] * b[i][obs[0]] * beta[0][i] for i in range(N))
    return beta, P


def baum_welch(observations, n_state, n_obs, max_iter=50):
    """Baum-Welch 算法学习 HMM 参数"""
    N, M = n_state, n_obs
    T = len(observations)

    # 1. 随机初始化参数 (确保每行和为1)
    random.seed(42)
    pi = [random.random() for _ in range(N)]
    pi = [v / sum(pi) for v in pi]

    a = [[random.random() for _ in range(N)] for _ in range(N)]
    a = [[a[i][j] / sum(a[i]) for j in range(N)] for i in range(N)]

    b = [[random.random() for _ in range(M)] for _ in range(N)]
    b = [[b[i][j] / sum(b[i]) for j in range(M)] for i in range(N)]

    for iteration in range(max_iter):
        # ---- E 步 ----
        alpha, P_fwd = forward(a, b, pi, observations)
        beta, P_bwd = backward(a, b, pi, observations)
        P = P_fwd  # P(O|λ)

        # 计算 γ_t(i)
        gamma = [[0] * N for _ in range(T)]
        for t in range(T):
            for i in range(N):
                gamma[t][i] = alpha[t][i] * beta[t][i] / P

        # 计算 ξ_t(i,j)
        xi = [[[0] * N for _ in range(N)] for _ in range(T - 1)]
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    xi[t][i][j] = (alpha[t][i] * a[i][j] * b[j][observations[t + 1]] * beta[t + 1][j]) / P

        # ---- M 步 ----
        # 更新 π
        pi = gamma[0][:]

        # 更新 A
        for i in range(N):
            for j in range(N):
                numerator = sum(xi[t][i][j] for t in range(T - 1))  # Σ ξ_t(i,j)
                denominator = sum(gamma[t][i] for t in range(T - 1))  # Σ γ_t(i)
                a[i][j] = numerator / denominator

        # 更新 B
        for j in range(N):
            for k in range(M):
                numerator = sum(gamma[t][j] for t in range(T) if observations[t] == k)  # Σ γ_t(j) [o_t=k]
                denominator = sum(gamma[t][j] for t in range(T))  # Σ γ_t(j)
                b[j][k] = numerator / denominator

        print(f"Iteration {iteration + 1}: P(O|λ) = {P:.6f}")

    return a, b, pi


if __name__ == "__main__":
    # 观测序列: 比如温度记录 (0=冷, 1=暖, 2=热)
    observations = [0, 0, 1, 0, 2, 1, 0]
    n_state = 2  # 隐藏状态数 (比如 2种天气模式)
    n_obs = 3  # 观测种类数

    a, b, pi = baum_welch(observations, n_state, n_obs, max_iter=20)

    print("\n学到的转移矩阵 A:")
    for row in a:
        print([f"{v:.4f}" for v in row])

    print("\n学到的发射矩阵 B:")
    for row in b:
        print([f"{v:.4f}" for v in row])

    print("\n学到的初始概率 π:")
    print([f"{v:.4f}" for v in pi])
