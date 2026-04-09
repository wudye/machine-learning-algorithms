"""
前向算法算的是 "所有路径的概率之和"（sum），Viterbi 算法算的是 "所有路径中的概率最大值"（max）。
前向算法:  α_t(i) = Σ_j [ α_{t-1}(j) · a_{ji} ] · b_i(o_t)    ← 所有可能路径求和
Viterbi:   δ_t(i) = max_j [ δ_{t-1}(j) · a_{ji} ] · b_i(o_t)   ← 只保留最优路径

graph TD
    A["初始化 dp[0][i] = π_i · b_i(o₁)"] --> B["递推 t=1..T:<br/>δ_t(i) = max_j [δ_{t-1}(j)·a_ji] · b_i(o_t)<br/>同时记录 last[t][i] = argmax_j"]
    B --> C["找终点: best_end = argmax_i δ_T(i)"]
    C --> D["回溯 t=T-1..1:<br/>ans[t] = last[t+1][ans[t+1]]"]
    D --> E["返回最优状态序列 ans"]

"""

def viterbi_algorithm(a, b, p, sequence):
    """维特比算法预测状态序列"""
    n_samples = len(sequence)
    n_state = len(a)  # 可能的状态数

    # 定义状态矩阵
    dp = [[0.0] * n_state for _ in range(n_samples)]  # 概率最大值
    last = [[-1] * n_state for _ in range(n_samples)]  # 上一个结点

    # 处理t=0的情况 (t=0) 时刻，每个状态的最大路径概率 = 初始概率 × 发射概率。
    """
    dp[0][0] = 0.2 × 0.5 = 0.10    last[0][0] = -1（起点，无前驱）
    dp[0][1] = 0.4 × 0.4 = 0.16    last[0][1] = -1
    dp[0][2] = 0.4 × 0.7 = 0.28    last[0][2] = -1

    """
    for i in range(n_state):
        dp[0][i] = p[i] * b[i][sequence[0]]

    """
    以 (t=1, i=1)（状态 1）为例：

    从 (j=0) 来：(0.10 \times 0.2 = 0.020)
    从 (j=1) 来：(0.16 \times 0.5 = 0.080)
    从 (j=2) 来：(0.28 \times 0.3 = 0.084) ← 最大！
    所以 (\delta_1(1) = 0.084)，(\text{last}[1][1] = 2)（最优前驱是状态 2）。
    
    再乘上发射概率：(\delta_1(1) = 0.084 \times b_1(1) = 0.084 \times 0.6 = 0.0504)
    """
    # 处理t>0的情况
    for t in range(1, n_samples):
        for i in range(n_state): # 当前状态
            for j in range(n_state): # 前一个状态
                delta = dp[t - 1][j] * a[j][i]
                if delta >= dp[t][i]:
                    dp[t][i] = delta # 留下最大概率
                    last[t][i] = j   # 记录前驱状态
            dp[t][i] *= b[i][sequence[t]] # 乘上发射概率

    # 计算最优路径的终点 在最后一个时刻 (t=T)，所有状态中概率最大的就是最优路径的终点。
    best_end, best_gamma = 0, 0
    for i in range(n_state):
        if dp[-1][i] > best_gamma:
            best_end, best_gamma = i, dp[-1][i]

    # 计算最优路径 从终点出发，沿着 last 指针反推每一步的最优前驱状态，最终得到完整路径。
    ans = [0] * (n_samples - 1) + [best_end]
    for t in range(n_samples - 1, 0, -1):
        ans[t - 1] = last[t][ans[t]]
    return ans


if __name__ == "__main__":
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    p = [0.2, 0.4, 0.4]
    print(viterbi_algorithm(A, B, p, [0, 1, 0]))  # [2, 2, 2]

"""
与近似算法的对比
          Viterbi（7viterbi.py）	近似算法（approximate.py）
每步选择	   max + 记录路径	max（独立选）
是否有回溯   有（last 数组）	无
考虑什么	   整条路径的联合概率	每个时刻局部概率
是否全局最优  是	            否
时间复杂度   (O(N^2 T))	    (O(N^2 T))
状态:   s1 → s3 → s1    概率 = 0.3
        s1 → s2 → s2    概率 = 0.25

近似算法可能输出: [s1, s3, s2]    ← t=2 选了 s2（局部最优），但这条路径不存在！
Viterbi 输出:       [s1, s3, s1]   ← 保证路径合法且全局最优

"""