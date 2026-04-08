
---

## 一、高斯混合模型（GMM）是什么？

假设数据由 $K$ 个高斯分布混合生成，每个数据点**不知道**来自哪个分布：

$$p(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x | \mu_k, \sigma_k^2)$$

其中：
- $\pi_k$：第 $k$ 个高斯的**混合系数**（权重），$\sum \pi_k = 1$
- $\mu_k$：第 $k$ 个高斯的**均值**
- $\sigma_k^2$：第 $k$ 个高斯的**方差**

```
数据 x: -3, -2.5, -1, 0.5, 1, 2, 3, 2.5

实际上由两个高斯生成：
  高斯1 (π₁=0.4, μ₁=-2, σ₁²=1)   ← 生成 -3, -2.5, -1
  高斯2 (π₂=0.6, μ₂= 2, σ₂²=1)   ← 生成  0.5, 1, 2, 3, 2.5

但我们不知道哪个点属于哪个高斯！
```

这就是一个**含有隐变量**的问题——隐变量 $z_i$ 表示"第 $i$ 个样本来自哪个高斯"。

---

## 二、为什么需要 EM 算法？

如果知道每个样本属于哪个高斯（$z_i$ 已知），那直接用最大似然估计：

$$\mu_k = \frac{\sum_{i: z_i=k} x_i}{\sum_{i: z_i=k} 1}, \quad \sigma_k^2 = \frac{\sum_{i: z_i=k} (x_i - \mu_k)^2}{\sum_{i: z_i=k} 1}$$

如果知道参数 $(\pi, \mu, \sigma^2)$，那可以推断每个样本属于哪个高斯：

$$\gamma_{ik} = P(z_i = k | x_i) = \frac{\pi_k \cdot \mathcal{N}(x_i | \mu_k, \sigma_k^2)}{\sum_{j=1}^{K} \pi_j \cdot \mathcal{N}(x_i | \mu_j, \sigma_j^2)}$$

**问题：两者都未知！** 这就是一个"鸡生蛋蛋生鸡"的问题 → 用 EM 算法迭代解决。

---

## 三、EM 算法推导

### E 步（Expectation）：计算责任度 $\gamma_{ik}$

给定当前参数，计算每个样本属于每个高斯的**概率**（责任度）：

$$\gamma_{ik} = \frac{\pi_k \cdot \mathcal{N}(x_i | \mu_k, \sigma_k^2)}{\sum_{j=1}^{K} \pi_j \cdot \mathcal{N}(x_i | \mu_j, \sigma_j^2)}$$

其中一维高斯密度函数：

$$\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

$\gamma_{ik}$ 的含义：样本 $i$ 有多大程度属于高斯 $k$。

### M 步（Maximization）：更新参数

用 $\gamma_{ik}$ 作为"软标签"，更新参数：

$$N_k = \sum_{i=1}^{N} \gamma_{ik} \quad \text{（第 k 个高斯的"有效样本数"）}$$

$$\mu_k = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} \cdot x_i$$

$$\sigma_k^2 = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} \cdot (x_i - \mu_k)^2$$

$$\pi_k = \frac{N_k}{N}$$

### 迭代流程

```
初始化: μ₁, μ₂, ..., μ_K, σ₁², σ₂², ..., σ_K², π₁, π₂, ..., π_K

重复直到收敛:
    E步: 计算责任度 γ_ik（每个样本属于每个高斯的概率）
    M步: 用 γ 更新 μ, σ², π
```

---

## 四、用数值例子走一遍

假设有 5 个数据点：$x = [-2, -1, 0, 1, 3]$，$K=2$

### 初始化

$$\mu_1 = -1, \quad \mu_2 = 1, \quad \sigma_1^2 = \sigma_2^2 = 1, \quad \pi_1 = \pi_2 = 0.5$$

### 第1轮 E 步

对 $x_1 = -2$：

$$\mathcal{N}(-2 | -1, 1) = \frac{1}{\sqrt{2\pi}} e^{-0.5} = 0.3521$$

$$\mathcal{N}(-2 | 1, 1) = \frac{1}{\sqrt{2\pi}} e^{-4.5} = 0.0044$$

$$\gamma_{11} = \frac{0.5 \times 0.3521}{0.5 \times 0.3521 + 0.5 \times 0.0044} = \frac{0.1761}{0.1783} = 0.988$$

$$\gamma_{12} = 1 - 0.988 = 0.012$$

> $x_1=-2$ 有 98.8% 的概率属于高斯1，1.2% 属于高斯2。符合直觉！

对每个点都这样算，得到责任度矩阵 $\gamma$（$N \times K$）。

### 第1轮 M 步

$$N_1 = \sum_i \gamma_{i1}, \quad N_2 = \sum_i \gamma_{i2}$$

$$\mu_1 = \frac{\gamma_{11} \times (-2) + \gamma_{21} \times (-1) + \dots}{N_1}$$

$$\sigma_1^2 = \frac{\gamma_{11} \times (-2-\mu_1)^2 + \dots}{N_1}$$

$$\pi_1 = \frac{N_1}{N}, \quad \pi_2 = \frac{N_2}{N}$$

### 重复...

参数逐步收敛到真实值。

---

## 五、完整代码实现

```python
import math
from collections import Counter
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.model_selection import train_test_split


def gaussian(x, mean, var):
    """一维高斯密度函数"""
    return (1 / math.sqrt(2 * math.pi * var)) * math.exp(-(x - mean) ** 2 / (2 * var))


class GaussianMixture:

    def __init__(self, X, init_mean, n_components, max_iter=100):
        self.X = X
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_samples = len(X)

        # 初始化参数
        self.means = init_mean.copy()               # 均值 μ
        self.variances = [1.0] * n_components        # 方差 σ²
        self.pis = [1.0 / n_components] * n_components  # 混合系数 π

        self.gamma = [[0] * n_components for _ in range(self.n_samples)]  # 责任度矩阵

        self._train()

    def _e_step(self):
        """E步: 计算责任度 γ_ik"""
        for i in range(self.n_samples):
            # 计算每个高斯对样本 i 的"贡献"
            weighted_probs = [
                self.pis[k] * gaussian(self.X[i], self.means[k], self.variances[k])
                for k in range(self.n_components)
            ]
            total = sum(weighted_probs)
            for k in range(self.n_components):
                self.gamma[i][k] = weighted_probs[k] / total

    def _m_step(self):
        """M步: 用责任度更新参数"""
        for k in range(self.n_components):
            # N_k: 第 k 个高斯的有效样本数
            N_k = sum(self.gamma[i][k] for i in range(self.n_samples))

            # 更新均值 μ_k
            self.means[k] = sum(self.gamma[i][k] * self.X[i] for i in range(self.n_samples)) / N_k

            # 更新方差 σ²_k
            self.variances[k] = sum(
                self.gamma[i][k] * (self.X[i] - self.means[k]) ** 2
                for i in range(self.n_samples)
            ) / N_k

            # 更新混合系数 π_k
            self.pis[k] = N_k / self.n_samples

    def _train(self):
        """EM 迭代"""
        for m in range(self.max_iter):
            old_means = self.means.copy()
            self._e_step()   # E步: 计算 γ
            self._m_step()   # M步: 更新 μ, σ², π
            # 收敛判断
            if all(abs(self.means[k] - old_means[k]) < 1e-6 for k in range(self.n_components)):
                print(f"EM 收敛于第 {m+1} 轮")
                break

    def predict(self, x):
        """预测: 返回概率最大的高斯编号"""
        probs = [
            self.pis[k] * gaussian(x, self.means[k], self.variances[k])
            for k in range(self.n_components)
        ]
        return probs.index(max(probs))
```

---

## 六、代码与公式对应总结

| 代码 | 数学公式 | 步骤 |
|:---|:---|:---:|
| `gaussian(x, mean, var)` | $\mathcal{N}(x \mid \mu_k, \sigma_k^2)$ | 高斯密度函数 |
| `weighted_probs[k]` | $\pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)$ | 加权概率 |
| `self.gamma[i][k]` | $\gamma_{ik}$ | **E步**：责任度 |
| `N_k = sum(gamma[i][k])` | $N_k = \sum_i \gamma_{ik}$ | 有效样本数 |
| `means[k] = sum(gamma*x) / N_k` | $\mu_k = \frac{1}{N_k}\sum_i \gamma_{ik} x_i$ | **M步**：更新均值 |
| `variances[k] = sum(gamma*(x-μ)²) / N_k` | $\sigma_k^2 = \frac{1}{N_k}\sum_i \gamma_{ik}(x_i-\mu_k)^2$ | **M步**：更新方差 |
| `pis[k] = N_k / n_samples` | $\pi_k = \frac{N_k}{N}$ | **M步**：更新混合系数 |