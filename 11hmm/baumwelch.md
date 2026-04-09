

## 一、Baum-Welch 是什么？

它是 **HMM 的 EM 算法**——给定一组观测序列，自动学习出最优参数 $\lambda = (A, B, \pi)$。

```
已知:  多组观测序列 O₁, O₂, ..., O_S
未知:  转移矩阵 A, 发射矩阵 B, 初始概率 π

Baum-Welch 帮你从数据中学出 A, B, π
```

和 GMM 的 EM 一样，也是 **E步 + M步** 迭代。

---

## 二、数学推导

### E 步：用当前参数计算 γ 和 ξ

**γ（单时刻状态概率）**：时刻 $t$ 处于状态 $i$ 的概率

$$\gamma_t(i) = P(q_t = i | O, \lambda) = \frac{\alpha_t(i) \cdot \beta_t(i)}{P(O|\lambda)}$$

**ξ（相邻时刻转移概率）**：时刻 $t$ 在 $i$，时刻 $t+1$ 在 $j$ 的概率

$$\xi_t(i,j) = P(q_t=i, q_{t+1}=j | O, \lambda) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{P(O|\lambda)}$$

### M 步：用 γ 和 ξ 更新参数

$$\pi_i = \gamma_1(i)$$

$$a_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

$$b_j(k) = \frac{\sum_{t=1, o_t=k}^{T} \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)}$$

---

## 三、用直觉理解三个更新公式

### 更新 π：第一时刻在状态 $i$ 的概率

```
π_i = γ₁(i)

"第1天最可能是什么天气？" → 这就是新的初始概率
```

### 更新 A：从 $i$ 转移到 $j$ 的比例

```
a_{ij} = Σ ξ_t(i,j) / Σ γ_t(i)
         ↑ 分子         ↑ 分母

分母 = 时刻 t 在状态 i 的所有情况 (不管转到哪里)
分子 = 时刻 t 在状态 i 且转到 j 的情况

分子/分母 = "在状态 i 时，转到 j 的比例" → 新的转移概率
```

```
比如统计10天的天气转移:
  状态i=晴 出现了 6 次 (Σγ_t(晴) = 6)
  其中 晴→晴 3次, 晴→阴 2次, 晴→雨 1次

  a(晴,晴) = 3/6 = 0.5
  a(晴,阴) = 2/6 = 0.33
  a(晴,雨) = 1/6 = 0.17
```

### 更新 B：在状态 $j$ 观测到 $k$ 的比例

```
b_j(k) = Σ γ_t(j) [o_t=k] / Σ γ_t(j)

分母 = 在状态 j 的所有时刻 (不管观测到什么)
分子 = 在状态 j 且观测到 k 的时刻

分子/分母 = "在状态 j 时，观测到 k 的比例" → 新的发射概率
```

```
比如统计在"雨天"的行为:
  雨天出现了 4 次
  其中 散步1次, 购物1次, 打扫2次

  b(雨,散步) = 1/4 = 0.25
  b(雨,购物) = 1/4 = 0.25
  b(雨,打扫) = 2/4 = 0.50
```

---

## 四、与 GMM EM 的对比

| | GMM 的 EM | HMM 的 Baum-Welch |
|:---|:---|:---|
| 隐变量 | $z_i$（样本属于哪个高斯） | $q_t$（时刻 $t$ 的隐藏状态） |
| E 步 | 计算 $\gamma_{ik}$（责任度） | 计算 $\gamma_t(i)$ 和 $\xi_t(i,j)$ |
| M 步更新 μ | $\mu_k = \frac{\sum \gamma_{ik} x_i}{\sum \gamma_{ik}}$ | π_i = γ₁(i) |
| M 步更新 σ² | $\sigma_k^2 = \frac{\sum \gamma_{ik}(x_i-\mu_k)^2}{\sum \gamma_{ik}}$ | $a_{ij} = \frac{\sum \xi_t(i,j)}{\sum \gamma_t(i)}$ |
| M 步更新 α | $\alpha_k = \frac{\sum \gamma_{ik}}{N}$ | $b_j(k) = \frac{\sum_{o_t=k} \gamma_t(j)}{\sum \gamma_t(j)}$ |
| 核心思想 | "这个样本多大程度属于这个高斯" | "这个时刻多大程度处于这个状态" |

本质完全一样：**E 步算软分配，M 步用软分配更新参数**。

---

## 五、完整代码实现

```python
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
            alpha[t][i] = sum(alpha[t-1][j] * a[j][i] for j in range(N)) * b[i][obs[t]]
    
    P = sum(alpha[T-1])
    return alpha, P

def backward(a, b, pi, obs):
    """后向算法, 返回所有时刻的 beta 和 P(O|λ)"""
    T = len(obs)
    N = len(a)
    beta = [[0] * N for _ in range(T)]
    
    # 初始化
    for i in range(N):
        beta[T-1][i] = 1.0
    
    # 递推 (从后往前)
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t][i] = sum(a[i][j] * b[j][obs[t+1]] * beta[t+1][j] for j in range(N))
    
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
        xi = [[[0] * N for _ in range(N)] for _ in range(T-1)]
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    xi[t][i][j] = (alpha[t][i] * a[i][j] * b[j][observations[t+1]] * beta[t+1][j]) / P
        
        # ---- M 步 ----
        # 更新 π
        pi = gamma[0][:]
        
        # 更新 A
        for i in range(N):
            for j in range(N):
                numerator = sum(xi[t][i][j] for t in range(T-1))   # Σ ξ_t(i,j)
                denominator = sum(gamma[t][i] for t in range(T-1)) # Σ γ_t(i)
                a[i][j] = numerator / denominator
        
        # 更新 B
        for j in range(N):
            for k in range(M):
                numerator = sum(gamma[t][j] for t in range(T) if observations[t] == k)  # Σ γ_t(j) [o_t=k]
                denominator = sum(gamma[t][j] for t in range(T))                       # Σ γ_t(j)
                b[j][k] = numerator / denominator
        
        print(f"Iteration {iteration+1}: P(O|λ) = {P:.6f}")
    
    return a, b, pi


if __name__ == "__main__":
    # 观测序列: 比如温度记录 (0=冷, 1=暖, 2=热)
    observations = [0, 0, 1, 0, 2, 1, 0]
    n_state = 2  # 隐藏状态数 (比如 2种天气模式)
    n_obs = 3    # 观测种类数
    
    a, b, pi = baum_welch(observations, n_state, n_obs, max_iter=20)
    
    print("\n学到的转移矩阵 A:")
    for row in a:
        print([f"{v:.4f}" for v in row])
    
    print("\n学到的发射矩阵 B:")
    for row in b:
        print([f"{v:.4f}" for v in row])
    
    print("\n学到的初始概率 π:")
    print([f"{v:.4f}" for v in pi])
```

---

## 六、整体流程图

```
┌──────────────────────────────────────────────────┐
│              Baum-Welch 算法                       │
│                                                    │
│  输入: 观测序列 O, 隐状态数 K, 观测种类数 M         │
│                                                    │
│  初始化: 随机 A, B, π                               │
│       ↓                                            │
│  ┌──────────────────────────────────────┐          │
│  │ E步:                                 │          │
│  │   ① 前向算法 → α                    │          │
│  │   ② 后向算法 → β                    │          │
│  │   ③ γ_t(i) = α_t(i)·β_t(i) / P(O)  │          │
│  │   ④ ξ_t(i,j) = α·a·b·β / P(O)      │          │
│  └──────────┬───────────────────────────┘          │
│             ↓ γ, ξ                                │
│  ┌──────────────────────────────────────┐          │
│  │ M步:                                 │          │
│  │   ① π_i = γ₁(i)                     │          │
│  │   ② a_{ij} = Σξ(i,j) / Σγ(i)       │          │
│  │   ③ b_j(k) = Σγ(j)[o=k] / Σγ(j)    │          │
│  └──────────┬───────────────────────────┘          │
│             ↓ 新 A, B, π                           │
│        重复迭代...                                 │
│             ↓                                      │
│  输出: 学到的 λ = (A, B, π)                        │
└──────────────────────────────────────────────────┘
```

---

## 七、为什么 P(O|λ) 每轮都递增？

Baum-Welch 是 EM 算法的特例，EM 保证**每轮迭代对数似然单调递增**：

```
P(O|λ) (对数似然)
  ↑
  │                ┌────── 收敛
  │              ──┘
  │            ──
  │          ──
  │        ──
  │      ──
  └────────────────────→ 迭代次数
  第1轮  第5轮  第15轮

每一轮的参数都比上一轮"更合理"
```

