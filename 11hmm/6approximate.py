"""
为什么近似算法不能得到全局最优？
核心原因只有一个：近似算法在每个时刻独立选最优状态，忽略了状态之间的连续性约束。
近似算法优化的是每个时刻边际概率（marginal probability）之和：

[ \arg\max_{i_1, i_2, \ldots, i_T} \sum_t \gamma_t(i_t) ]

这是在最大化独立决策的正确率，但完全不在乎相邻状态之间能不能转移过去。

Viterbi 优化的是整条路径的联合概率（joint probability）：

[ \arg\max_{i_1, i_2, \ldots, i_T} P(q_1=i_1, q_2=i_2, \ldots, q_T=i_T \mid O, \lambda) ]

这是在最大化整条路径的概率，每一步都确保转移是合法的。
近似算法 = 贪心策略
┌──────────────────────────────────────────────┐
│  你在开车导航，每个路口都选"离终点直线距离最近"的路 │
│  结果可能开进死胡同，因为没考虑路的连通性         │
└──────────────────────────────────────────────┘

Viterbi = 全局动态规划
┌──────────────────────────────────────────────┐
│  你在开车导航，每一步都考虑"从起点到当前点的最优路径" │
│  最后回溯，保证从起点到终点整条路都是最优的        │
└──────────────────────────────────────────────┘

even (\gamma_t(i) = \alpha_t(i) \times \beta_t(i)) 确实考虑了状态转移。问题不在于 (\gamma) 没考虑转移，
而在于 (\gamma) 是一个 边际概率（marginal probability），它对所有路径求和了
假设 3 个时刻，2 个状态（s0, s1），转移矩阵强烈偏好保持不变：
A = [[0.9, 0.1],    ← s0 有 90% 概率留在 s0
     [0.1, 0.9]]    ← s1 有 90% 概率留在 s1
假设算出来 (\gamma) 值如下（数字只是为了说明问题的假设值）：
时刻	(\gamma(s0))	(\gamma(s1))	近似算法选择
t=0	0.51	0.49	s0
t=1	0.49	0.51	s1
t=2	0.51	0.49	s0
输出：[s0, s1, s0]

(\gamma_1(s1) = 0.51) 这个值是怎么来的？它是所有经过 s1 的路径的概率之和
γ₁(s1) = P(q₀=s0, q₁=s1, q₂=s0 | O)     ← 路径 A: s0→s1→s0
       + P(q₀=s0, q₁=s1, q₂=s1 | O)     ← 路径 B: s0→s1→s1
       + P(q₀=s1, q₁=s1, q₂=s0 | O)     ← 路径 C: s1→s1→s0
       + P(q₀=s1, q₁=s1, q₂=s1 | O)     ← 路径 D: s1→s1→s1
       = 0.51

这 0.51 是四条路径加在一起的结果！但近似算法在 t=1 选了 s1，在 t=0 选了 s0，在 t=2 又选了 s0，最终拼出的是路径 A：
P(s0→s1→s0) = ... × 0.1 × ... × 0.1 × ...    ← 转移概率极低！
而真正的全局最优路径可能是路径 D：
P(s1→s1→s1) = ... × 0.9 × ... × 0.9 × ...    ← 转移概率很高！

(\gamma_1(s1) = 0.51) 是一个"混合"的概率，不是某一条具体路径的概率。 你把不同时刻的最大 (\gamma) 拼起来，等于从不同的路径里各取了一段，拼成了一条根本不存在的路径。
t=0 选 s0 主要因为路径 s0,s0,s0 贡献大
t=1 选 s1 主要因为路径 s0,s1,s1 贡献大
t=2 选 s0 主要因为路径 s0,s0,s0 贡献大
但拼在一起 s0,s1,s0 既不是 s0,s0,s0，也不是 s0,s1,s1——它是一条被"杂交"出来的不存在的路径
"""

def approximation_algorithm(a, b, p, sequence):
    """近似算法预测状态序列"""
    n_samples = len(sequence)
    n_state = len(a)  # 可能的状态数

    # ---------- 计算：前向概率 ----------
    # 计算初值（定义状态矩阵）
    dp = [p[i] * b[i][sequence[0]] for i in range(n_state)]
    alpha = [dp]

    # 递推（状态转移）
    for t in range(1, n_samples):
        dp = [sum(a[j][i] * dp[j] for j in range(n_state)) * b[i][sequence[t]] for i in range(n_state)]
        alpha.append(dp)

    # ---------- 计算：后向概率 ----------
    # 计算初值（定义状态矩阵）
    dp = [1] * n_state
    beta = [dp]

    # 递推（状态转移）
    for t in range(n_samples - 1, 0, -1):
        dp = [sum(a[i][j] * dp[j] * b[j][sequence[t]] for j in range(n_state)) for i in range(n_state)]
        beta.append(dp)

    beta.reverse()

    # 计算最优可能的状态序列
    ans = []
    for t in range(n_samples):
        min_state, min_gamma = -1, 0
        for i in range(n_state):
            gamma = alpha[t][i] * beta[t][i]
            if gamma > min_gamma:
                min_state, min_gamma = i, min_gamma
        ans.append(min_state)
    return ans


if __name__ == "__main__":
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    p = [0.2, 0.4, 0.4]
    print(approximation_algorithm(A, B, p, [0, 1, 0]))  # [2, 2, 2]