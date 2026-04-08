            # M
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

            self.alpha = alpha_new
            self.means = means_new
            self.sigma = sigma_new
这就是 EM 算法的 **M 步（Maximization）**，用 E 步算出的 $\gamma$ 矩阵来更新三个参数。

---

## 三个参数更新的对应公式

$$\mu_k = \frac{\sum_{j=1}^{N} \gamma_{jk} \cdot x_j}{\sum_{j=1}^{N} \gamma_{jk}}$$

$$\sigma_k^2 = \frac{\sum_{j=1}^{N} \gamma_{jk} \cdot (x_j - \mu_k)^2}{\sum_{j=1}^{N} \gamma_{jk}}$$

$$\alpha_k = \frac{\sum_{j=1}^{N} \gamma_{jk}}{N}$$

---

## 1. 更新均值 $\mu_k$

```36:43:h:\machine learning algorithm\machine-learning-algorithms\10em\em.py
            means_new = [0.0] * self.n_components
            for k in range(self.n_components):
                sum1 = 0  # 分子
                sum2 = 0  # 分母
                for j in range(self.n_samples):
                    sum1 += gamma[j][k] * self.X[j]
                    sum2 += gamma[j][k]
                means_new[k] = sum1 / sum2
```

| 代码 | 数学 | 含义 |
|:---|:---|:---|
| `sum1 += gamma[j][k] * self.X[j]` | $\sum_j \gamma_{jk} \cdot x_j$ | $\gamma_{jk}$ 加权后的样本值之和 |
| `sum2 += gamma[j][k]` | $\sum_j \gamma_{jk} = N_k$ | 第 $k$ 个高斯的"有效样本数" |
| `means_new[k] = sum1 / sum2` | $\mu_k = \frac{\sum_j \gamma_{jk} x_j}{N_k}$ | **加权平均** |

> 直觉：$\gamma_{jk}$ 是样本 $j$ 属于高斯 $k$ 的概率，所以 $\mu_k$ 就是所有样本的**概率加权平均值**——$\gamma_{jk}$ 越大，$x_j$ 对 $\mu_k$ 的贡献越大。

```
比如 5 个样本，高斯1 的 gamma 分别是 [0.9, 0.8, 0.1, 0.05, 0.02]

μ₁ = (0.9×(-2) + 0.8×(-1) + 0.1×0 + 0.05×1 + 0.02×3) / (0.9+0.8+0.1+0.05+0.02)
   = (-1.8 + (-0.8) + 0 + 0.05 + 0.06) / 1.87
   = -2.49 / 1.87
   = -1.33

x=-2 和 x=-1 的 gamma 很大，所以 μ₁ 被拉到负数方向 ← 合理！
```

---

## 2. 更新方差 $\sigma_k^2$

```45:52:h:\machine learning algorithm\machine-learning-algorithms\10em\em.py
            sigma_new = [1.0] * self.n_components
            for k in range(self.n_components):
                sum1 = 0  # 分子
                sum2 = 0  # 分母
                for j in range(self.n_samples):
                    sum1 += gamma[j][k] * math.pow(self.X[j] - self.means[k], 2)
                    sum2 += gamma[j][k]
                sigma_new[k] = sum1 / sum2
```

| 代码 | 数学 | 含义 |
|:---|:---|:---|
| `(self.X[j] - self.means[k])²` | $(x_j - \mu_k)^2$ | 样本与均值的偏差平方 |
| `gamma[j][k] * (x_j - μ_k)²` | $\gamma_{jk}(x_j - \mu_k)^2$ | 概率加权的偏差 |
| `sigma_new[k] = sum1 / sum2` | $\sigma_k^2 = \frac{\sum_j \gamma_{jk}(x_j-\mu_k)^2}{N_k}$ | **加权方差** |

> 直觉：和普通方差的公式 $\frac{1}{N}\sum(x_i - \bar{x})^2$ 一样，只是把"计数"换成了"概率权重" $\gamma$。

---

## 3. 更新混合系数 $\alpha_k$

```54:59:h:\machine learning algorithm\machine-learning-algorithms\10em\em.py
            alpha_new = [1 / self.n_components] * self.n_components
            for k in range(self.n_components):
                sum1 = 0  # 分子
                for j in range(self.n_samples):
                    sum1 += gamma[j][k]
                alpha_new[k] = sum1 / self.n_samples
```

| 代码 | 数学 | 含义 |
|:---|:---|:---|
| `sum1 += gamma[j][k]` | $\sum_j \gamma_{jk} = N_k$ | 高斯 $k$ 的有效样本数 |
| `alpha_new[k] = sum1 / n_samples` | $\alpha_k = \frac{N_k}{N}$ | 高斯 $k$ 占总体的比例 |

> 直觉：如果所有样本"主要属于"高斯1（$\gamma_{j1}$ 大），那 $\alpha_1$ 就大，说明高斯1贡献的数据多。

---

## 最后：用新参数替换旧参数

```61:63:h:\machine learning algorithm\machine-learning-algorithms\10em\em.py
            self.alpha = alpha_new
            self.means = means_new
            self.sigma = sigma_new
```

替换后回到下一轮 E 步，用新参数重新计算 $\gamma$，如此迭代。

---

## EM 整体流程图

```
初始化: μ, σ², α
    │
    ▼
┌─────────────────────────────────────┐
│  E步: 用当前(μ,σ²,α)计算γ矩阵       │
│  "每个样本有多大可能属于哪个高斯？"    │
└──────────────┬──────────────────────┘
               │ γ 矩阵
               ▼
┌─────────────────────────────────────┐
│  M步: 用γ更新(μ,σ²,α)               │
│  "根据责任度重新估计参数"             │
│                                     │
│  μ_k  = Σγ·x / Σγ       (加权平均)  │
│  σ²_k = Σγ·(x-μ)² / Σγ  (加权方差)  │
│  α_k  = Σγ / N            (占比)    │
└──────────────┬──────────────────────┘
               │ 新参数
               ▼
            重复迭代...
               │
               ▼
            参数收敛 ✓
```

**E 步和 M 步互相配合**：E 步回答"样本属于哪个高斯"，M 步回答"那高斯应该长什么样"，两者交替直到参数不再变化。