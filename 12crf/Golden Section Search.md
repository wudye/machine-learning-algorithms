The user wants me to explain the golden section search algorithm.
## 黄金分割法（Golden Section Search）

### 一句话总结

在区间 $[a, b]$ 上找一个**一元函数的极小值点**，每次缩小搜索范围，直到区间足够小。

---

### 核心思想：每次只算一次函数值

普通做法：每次取中点，需要算 **2 个**新点的函数值。

黄金分割法：利用比例关系，每次只需要算 **1 个**新点的函数值，另一个点直接**复用上轮的结果**。

```
黄金比例: φ = 0.382, 1 - φ = 0.618

[a0]────────[a1]──────────[b1]────────[b0]
     38.2%       61.8%
```

---

### 逐行解释

#### 初始分割（第 17-18 行）

```17:18:h:\machine learning algorithm\machine-learning-algorithms\12crf\3Quasi-Newton Methods.py
    a1, b1 = a0 + 0.382 * (b0 - a0), b0 - 0.382 * (b0 - a0)
    fa, fb = func(a1), func(b1)
```

在 $[a_0, b_0]$ 内取两个试探点：

```
[a0]──────[a1]──────────────[b1]──────[b0]
  |← 0.382 →|←── 0.236 ──→|← 0.382 →|
```

---

#### 迭代缩小（第 20-28 行）

```20:28:h:\machine learning algorithm\machine-learning-algorithms\12crf\3Quasi-Newton Methods.py
    while b1 - a1 > epsilon:
        if fa <= fb:
            b0, b1, fb = b1, a1, fa
            a1 = a0 + 0.382 * (b0 - a0)
            fa = func(a1)
        else:
            a0, a1, fa = a1, b1, fb
            b1 = b0 - 0.382 * (b0 - a0)
            fb = func(b1)
```

**情况 1**：$f(a_1) \leq f(b_1)$ → 极小值在左半部分

```
情况1: fa <= fb → 丢弃 [b1, b0]，搜索 [a0, b1]

[a0]──────[a1]──────────────[b1]  [b0]
  保留        ↑变成新的b1      ↑变成新的b0  丢弃

新的区间: [a0, 新b0]
┌─────────────────────────────────┐
│ [a0]────[新a1]────────[新b1]    │  新a1 是新算的，新b1 = 旧a1（复用！）
└─────────────────────────────────┘
```

- $b_0$ 移到 $b_1$，$b_1$ 移到 $a_1$ → 区间缩小到右侧的 61.8%
- 只需新算一个点 `a1`，`b1` 和 `fb` 直接复用旧的 `a1` 和 `fa`

**情况 2**：$f(a_1) > f(b_1)$ → 极小值在右半部分

```
情况2: fa > fb → 丢弃 [a0, a1]，搜索 [a1, b0]

[a0]  [a1]──────────────[b1]──────[b0]
 丢弃   ↑变成新的a0      ↑变成新的a1  保留

新的区间: [新a0, b0]
┌─────────────────────────────────┐
│      [新a1]────────[新b1]────[b0]│  新b1 是新算的，新a1 = 旧b1（复用！）
└─────────────────────────────────┘
```

---

#### 收敛返回（第 30 行）

```30:30:h:\machine learning algorithm\machine-learning-algorithms\12crf\3Quasi-Newton Methods.py
    return (a1 + b1) / 2
```

当 $b_1 - a_1 < \epsilon$ 时，区间已经足够小，取中点作为极小值点。

---

### 图解完整过程

以函数 $f(x) = (x-3)^2$ 在 $[0, 10]$ 上找极小值为例：

```
初始:  [0]──────[3.82]──────────────[6.18]──────[10]
       f(3.82)=0.67                f(6.18)=10.0
       fa < fb → 丢弃右半边

第1轮: [0]──────[2.36]──────[3.82]  6.18]  10
       f(2.36)=0.41  f(3.82)=0.67
       fa < fb → 丢弃右半边

第2轮: [0]──[1.46]──────[2.36]  3.82]
       f(1.46)=2.37  f(2.36)=0.41
       fa > fb → 丢弃左半边

...继续缩小...

最终:  ≈ 3.0（真正的极小值点）
```

---

### 在 BFGS 中的作用

在 BFGS 中，黄金分割法用来求解**一维最优步长**：

```247:252:h:\machine learning algorithm\machine-learning-algorithms\12crf\3Quasi-Newton Methods.py
    def f(xx):
        x2 = [w0[jj] + xx * pk[jj, 0].item() for jj in range(n_features)]
        return func(x2)

    lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)
```

BFGS 确定了搜索方向 $p_k$ 后，需要找到最优的步长 $\lambda$：

$$w_{k+1} = w_k + \lambda \cdot p_k$$

黄金分割法在 $[0, 20]$ 上搜索，找到让目标函数最小的 $\lambda$。