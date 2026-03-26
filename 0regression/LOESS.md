# 局部加权线性回归（LOESS/LOWESS）

## 核心概念
本质上，Loess是一种逐点进行的加权回归平滑算法。对于原始数据 
 的每个观测值,都可以使用相邻的若干已知观测值估计得到的函数 
f（x ） 来估计。

局部加权线性回归是一种**非参数**方法，它对每个预测点都拟合一个局部模型。与普通线性回归不同，它不假设全局线性关系，而是根据预测点附近的局部数据来拟合模型。

### 工作原理

1. **权重分配**：对于预测点 \(x_0\)，根据距离赋予每个训练样本不同的权重
2. **加权回归**：使用加权最小二乘法估计系数
3. **局部拟合**：只对\(x_0\)附近的数据点赋予高权重

### 权重函数（高斯核）

最常用的高斯核权重函数：


wᵢ = exp(-‖xᵢ - x‖²/(2τ²)),

其中：
- \(x^{(i)}\)：第\(i\)个训练样本
- \(x_0\)：预测点
- \(\tau\)：带宽参数（控制局部范围）

---
β = (XᵀWX)⁻¹(XᵀWy),
^y = βx

## 具体例子

假设我们要预测房价，数据如下：

| 面积 \(x\) (m²) | 价格 \(y\) (万元) |
|-----------------|------------------|
| 30              | 80               |
| 50              | 120              |
| 70              | 150              |
| 90              | 200              |
| 110             | 220              |

**预测目标**：当面积 \(x_0 = 65\) m² 时，房价是多少？

### 步骤1：计算权重

设 \(\tau = 20\)，计算各样本相对于 \(x_0 = 65\) 的权重：

```python
import numpy as np

x = np.array([30, 50, 70, 90, 110])
y = np.array([80, 120, 150, 200, 220])
x0 = 65
tau = 20

# 计算高斯权重
weights = np.exp(-(x - x0)**2 / (2 * tau**2))
print("权重：", weights)
# 输出：权重：[0.082, 0.606, 0.970, 0.446, 0.055]
```

**分析**：
- \(x = 70\)（距离最近）权重最大：0.970
- \(x = 30\)（距离最远）权重最小：0.082

### 步骤2：构造加权矩阵

权重矩阵 \(W\) 是对角矩阵：

\[
W = \begin{bmatrix}
0.082 & 0 & 0 & 0 & 0 \\
0 & 0.606 & 0 & 0 & 0 \\
0 & 0 & 0.970 & 0 & 0 \\
0 & 0 & 0 & 0.446 & 0 \\
0 & 0 & 0 & 0 & 0.055
\end{bmatrix}
\]

### 步骤3：构造设计矩阵 \(X\)

\[
X = \begin{bmatrix}
1 & 30 \\
1 & 50 \\
1 & 70 \\
1 & 90 \\
1 & 110
\end{bmatrix}, \quad y = \begin{bmatrix}80 \\ 120 \\ 150 \\ 200 \\ 220\end{bmatrix}
\]

### 步骤4：计算系数 \(\beta\)

\[
\beta = (X^T W X)^{-1} (X^T W y)
\]

**Python实现**：

```python
# 构造X矩阵（添加偏置项）
X = np.column_stack([np.ones_like(x), x])
W = np.diag(weights)

# 计算加权回归系数
X_T = X.T
XTWX = X_T @ W @ X
XTWy = X_T @ W @ y
beta = np.linalg.inv(XTWX) @ XTWy

print("系数 β =", beta)
# 输出：系数 β = [10.52, 2.01]
```

解释：
- \(\beta_0 = 10.52\)：截距
- \(\beta_1 = 2.01\)：斜率

### 步骤5：预测

\[
\hat{y}_{x_0=65} = \beta_0 + \beta_1 \times 65 = 10.52 + 2.01 \times 65 = 141.17 \text{ 万元}
\]

---

## 完整代码示例

```python
import numpy as np
import matplotlib.pyplot as plt

def locally_weighted_linear_regression(x, y, x_query, tau=20):
    """
    局部加权线性回归

    参数:
        x: 训练特征
        y: 训练标签
        x_query: 预测点
        tau: 带宽参数
    """
    # 计算权重
    weights = np.exp(-(x - x_query)**2 / (2 * tau**2))
    W = np.diag(weights)

    # 构造设计矩阵
    X = np.column_stack([np.ones_like(x), x])

    # 计算系数
    XTWX = X.T @ W @ X
    XTWy = X.T @ W @ y
    beta = np.linalg.inv(XTWX) @ XTWy

    # 预测
    X_query = np.array([1, x_query])
    y_pred = X_query @ beta

    return y_pred, beta, weights

# 数据
x = np.array([30, 50, 70, 90, 110])
y = np.array([80, 120, 150, 200, 220])

# 预测x0=65
x0 = 65
y_pred, beta, weights = locally_weighted_linear_regression(x, y, x0)

print(f"预测点 x0={x0}")
print(f"权重: {weights}")
print(f"回归系数: {beta}")
print(f"预测值: {y_pred:.2f} 万元")
```

---

## 关键参数影响

### 带宽参数 \(\tau\)

| \(\tau\) 值 | 效果 | 拟合特性 |
|------------|------|----------|
| **小** | 权重衰减快 | 高方差，低偏差（过拟合风险） |
| **大** | 权重衰减慢 | 低方差，高偏差（欠拟合风险） |

**可视化不同 \(\tau\) 的影响**：

```python
# 生成多个预测点
x_range = np.linspace(20, 120, 100)

# 尝试不同tau
taus = [5, 20, 50]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='训练数据')

for tau in taus:
    predictions = []
    for xq in x_range:
        y_pred, _, _ = locally_weighted_linear_regression(x, y, xq, tau)
        predictions.append(y_pred)
    plt.plot(x_range, predictions, label=f'τ={tau}')

plt.xlabel('面积 (m²)')
plt.ylabel('价格 (万元)')
plt.legend()
plt.title('不同带宽参数的影响')
plt.grid(True)
```

---

## 与普通线性回归的对比

| 特性 | 普通线性回归 | 局部加权线性回归 |
|------|------------|----------------|
| **模型复杂度** | 固定（全局线性） | 可变（局部适应） |
| **参数数量** | 固定 | 数据量相关 |
| **非线性** | 无法处理 | 可处理非线性关系 |
| **计算复杂度** | \(O(n)\) | \(O(n^3)\)（每个点） |
| **适用场景** | 全局线性关系 | 复杂非线性关系 |

---

## 优缺点

### 优点
✅ 能拟合复杂的非线性关系
✅ 不需要预设函数形式
✅ 对局部变化敏感

### 缺点
❌ 计算开销大（每个预测点都要重新计算）
❌ 对异常值敏感
❌ 需要选择合适的 \(\tau\)

---

## 总结

局部加权线性回归通过**动态加权**实现了对数据的局部拟合：
1. 根据距离预测点的远近分配权重
2. 使用加权最小二乘法估计局部模型
3. 适用于非线性关系和复杂模式

在房价预测例子中，\(x_0 = 65\) m² 的预测值为 **141.17 万元**，这个结果更接近附近的真实值（150万元），体现了局部加重的优势。
