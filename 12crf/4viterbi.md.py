def viterbi_algorithm(w1, transfer_features, w2, state_features, x, n_state):
    """维特比算法预测状态序列

    :param w1: 模型的转移特征权重
    :param t: 模型的转移特征函数
    :param w2: 模型的状态特征权重
    :param s: 模型的状态特征函数
    :param x: 需要计算的观测序列
    :param n_state: 状态的可能取值数
    :return: 最优可能的状态序列
    """
    n_transfer_features = len(transfer_features)  # 转移特征数
    n_state_features = len(state_features)  # 状态特征数
    n_position = len(x)  # 序列中的位置数

    # 定义状态矩阵
    dp = [[0.0] * n_state for _ in range(n_position)]  # 概率最大值
    last = [[-1] * n_state for _ in range(n_position)]  # 上一个结点
    print(dp)

    # 处理t=0的情况
    """
    位置 0 没有前驱状态，所以只计算状态特征：
    位置 0:
      δ₀(0) = w2[0]·s1(0,x,0) + w2[1]·s2(0,x,0) + w2[2]·s3(0,x,0) + w2[3]·s4(0,x,0)
      δ₀(1) = w2[0]·s1(1,x,0) + w2[1]·s2(1,x,0) + w2[2]·s3(1,x,0) + w2[3]·s4(1,x,0)
    (y0=i, x=x, i=0)
    def s1(y0, x, i):
        return int(y0 == 0 and i in {0})
    """
    for i in range(n_state):
        for l in range(n_state_features):
            dp[0][i] += w2[l] * state_features[l](y0=i, x=x, i=0)
    print(dp)

    # 处理t>0的情况
    """
    递推 t=1,2,...
                    ┌──────────────────────────────────────────┐
                │  δₜ(j) = max over i of:                  │
                │                                          │
                │  δₜ₋₁(i)     ← 到达前一个状态 i 的最佳得分  │
                │  + Σ wₖ·tₖ(i→j)  ← 从 i 转移到 j 的得分    │
                │  + Σ wₗ·sₗ(j)    ← 在 j 的状态得分          │
                └──────────────────────────────────────────┘
    用具体例子（n_state=2，位置 t=1）：
                      状态0              状态1
                 ┌─────┐           ┌─────┐
      δ₀(0)=1.5 ──→ │ ?   │  转移+状态  │ ?   │ ←── δ₀(1)=0.8
                     └─────┘           └─────┘
                     
      计算 dp[1][0]:
        从 i=0 来: d = 1.5 + 转移(0→0) + 状态(0) = 2.3
        从 i=1 来: d = 0.8 + 转移(1→0) + 状态(0) = 1.6
        max = 2.3 → dp[1][0] = 2.3, last[1][0] = 0
        
      计算 dp[1][1]:
        从 i=0 来: d = 1.5 + 转移(0→1) + 状态(1) = 2.8
        从 i=1 来: d = 0.8 + 转移(1→1) + 状态(1) = 1.5
        max = 2.8 → dp[1][1] = 2.8, last[1][1] = 0
    """
    for t in range(1, n_position):
        for i in range(n_state):
            for j in range(n_state):
                d = dp[t - 1][i]
                for k in range(n_transfer_features):
                    d += w1[k] * transfer_features[k](y0=i, y1=j, x=x, i=t)
                for l in range(n_state_features):
                    d += w2[l] * state_features[l](y0=j, x=x, i=t)
                # print((i, j), "=", d)
                if d >= dp[t][j]:
                    dp[t][j] = d
                    last[t][j] = i
        # print(dp[t], last[t])

    # 计算最优路径的终点
    """
    在最后一个位置，选得分最大的状态作为终点。
    位置 2:  dp[2][0] = 3.1,  dp[2][1] = 2.5
                      ↓
         best_end = 0,  best_gamma = 3.1
    """
    best_end, best_gamma = 0, 0
    for i in range(n_state):
        if dp[-1][i] > best_gamma:
            best_end, best_gamma = i, dp[-1][i]

    # 计算最优路径
    """
    从终点反向回溯，利用 last 数组找每一步的最佳前驱。
    回溯过程:
      t=2: ans[2] = 0 (best_end)
      t=2: ans[1] = last[2][0] = 1  → 前一个状态是 1
      t=1: ans[0] = last[1][1] = 0  → 前一个状态是 0
      
    最优路径: [0, 1, 0]
                状态 0       状态 1
             ┌──────────┬──────────┐
      t=0    │  δ=1.5   │  δ=0.8   │  ← 初始化（只有状态特征）
             │  last=-  │  last=-  │
             ├──────────┼──────────┤
      t=1    │  δ=2.3   │  δ=2.8   │  ← δ₀(0)+转移(0→0)+状态(0) = 2.3
             │  last=0  │  last=0  │     δ₀(0)+转移(0→1)+状态(1) = 2.8 ← max
             ├──────────┼──────────┤
      t=2    │  δ=3.1   │  δ=2.5   │  ← δ₁(1)+转移(1→0)+状态(0) = 3.1 ← max
             │  last=1  │  last=1  │
             └──────────┴──────────┘
                        ↑
                  best_end = 0
    
    回溯:  t=2→0,  t=1→1,  t=0→0
    最优路径: [0, 1, 0]
    """
    ans = [0] * (n_position - 1) + [best_end]
    for t in range(n_position - 1, 0, -1):
        ans[t - 1] = last[t][ans[t]]
    return ans


if __name__ == "__main__":
    import random

    def t1(y0, y1, x, i):
        return int(y0 == 0 and y1 == 1 and i in {1, 2})

    def t2(y0, y1, x, i):
        return int(y0 == 0 and y1 == 0 and i in {1})

    def t3(y0, y1, x, i):
        return int(y0 == 1 and y1 == 0 and i in {2})

    def t4(y0, y1, x, i):
        return int(y0 == 1 and y1 == 0 and i in {1})

    def t5(y0, y1, x, i):
        return int(y0 == 1 and y1 == 1 and i in {2})

    def s1(y0, x, i):
        return int(y0 == 0 and i in {0})

    def s2(y0, x, i):
        return int(y0 == 1 and i in {0, 1})

    def s3(y0, x, i):
        return int(y0 == 0 and i in {1, 2})

    def s4(y0, x, i):
        return int(y0 == 1 and i in {2})

    w1 = [1, 0.6, 1, 1, 0.2]
    t = [t1, t2, t3, t4, t5]
    w2 = [1, 0.5, 0.8, 0.5]
    s = [s1, s2, s3, s4]

    print(viterbi_algorithm(w1, t, w2, s, [random.randint(0, 1) for _ in range(3)], 2))  # [0, 1, 0]