import bisect
import math
import random

"""
问题设定
观测序列 x 和 状态序列 y 都是长度为 3 的二元组，取值空间均为：
    [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
    [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])

共 8 种可能。


"""

def count_conditional_probability(w1, t, w2, s, x, y):
    """已知条件随机场模型计算状态序列关于观测序列的非规范化条件概率

    :param w1: 模型的转移特征权重
    :param t: 模型的转移特征函数
    :param w2: 模型的状态特征权重
    :param s: 模型的状态特征函数
    :param x: 需要计算的观测序列
    :param y: 需要计算的状态序列
    :return: 状态序列关于观测序列的条件概率
    """
    n_features_1 = len(w1)  # 转移特征数
    n_features_2 = len(w2)  # 状态特征数
    n_position = len(x)  # 序列中的位置数

    res = 0
    for k in range(n_features_1):
        for i in range(1, n_position):
            res += w1[k] * t[k](y[i - 1], y[i], x, i)
    for k in range(n_features_2):
        for i in range(n_position):
            res += w2[k] * s[k](y[i], x, i)
    return pow(math.e, res)


def make_hidden_sequence(w1, t, w2, s, x_range, y_range, n_samples=1, random_state=0):
    """已知模型构造随机样本集

    :param w1: 模型的转移特征权重
    :param t: 模型的转移特征函数
    :param w2: 模型的状态特征权重
    :param s: 模型的状态特征函数
    :param x_range: 观测序列的可能取值
    :param y_range: 状态序列的可能取值
    :param n_samples: 生成样本集样本数(近似)
    :return: 状态序列关于观测序列的条件概率
    """
    P = [[0.0] * len(y_range) for _ in range(len(x_range))]  # 条件概率分布
    # print(len(y_range), len(x_range))


    """
    共有 8 个 x × 8 个 y = 64 对组合
    每对计算非规范化概率 exp(score)，四舍五入到 1 位小数
    lst 是这 64 个值的累积和
    把 lst 放到数轴上，每一格代表一个 (x,y) 对的"领地"：
    0 ────────────────────────────────────────────────────────────────── 7050.7
     ├─1.0─┤──2.7──┤──9.0──┤──11.0─┤─1.2─┤─3.3─┤─4.1─┤─5.0─┤ ...
     
     lst[0] lst[1]  lst[2]  lst[3]  ...                  ...      lst[63]

    你的 64 个值可以分成 8 组（每组 8 个），对应 8 个不同的 x:
    组	    x 值	累积值范围	    该组总概率	含义
    第1组	(0,0,0)	1.0 → 37.3	    37.3	所有 y 的概率之和
    第2组	(0,0,1)	38.3 → 54.2 	16.9	概率较低
    第3组	(0,1,0)	70.6 → 187.3	133.1	概率较高
    第4组	(0,1,1)	1823.3 → 2865.8	2683.0	非常高！
    第5组	(1,0,0)	3047.1 → 3248.4	201.3	
    第6组	(1,0,1)	6229.4 → 6949.2	719.8	高
    第7组	(1,1,0)	6979.2 → 7024.3	45.1	
    第8组	(1,1,1)	7029.3 → 7050.7	21.4	
    关键观察：差异巨大
    最大的单格跳跃出现在 第4组第1个值：从 187.3 跳到 1823.3，单格概率 ~1636
    这意味着 x=(0,1,1) 配合某个特定 y 的概率极其高
    哪个 (x,y) 对概率最高？
    第4组的第1个值（索引 24）对应 x=(0,1,1), y=(0,0,0)，它的非规范化概率约为：1823.3−187.3=1636.0
    占总概率的1636/7050.7≈23.2%，远超平均值 1/64≈1.56%
    """
    lst = []
    sum_ = 0
    for i, x in enumerate(x_range):
        # print("x", x)
        for j, y in enumerate(y_range):
            # print("y", y)
            P[i][j] = round(count_conditional_probability(w1, t, w2, s, x, y), 1)
            sum_ += P[i][j]
            lst.append(sum_)

    """
    概率越高的格子在数轴上占的空间越大，随机数落入其中的概率就越大——这就是按概率分布采样的核心思想。
    """
    X, Y = [], []
    # print(sum_) # 7050.699999999999
    print(lst)
    random.seed(random_state)
    for _ in range(n_samples):
        r = random.uniform(0, sum_)
        idx = bisect.bisect_left(lst, r)
        """
        len(y_range) = 8
        idx = 24
        
        divmod(24, 8) → (3, 0)
        
        i = 3  → x_range 的第 3 个元素
        j = 0  → y_range 的第 0 个元素
        """
        i, j = divmod(idx, len(y_range))
        X.append(x_range[i])
        Y.append(y_range[j])

    return X, Y


if __name__ == "__main__":
    # ---------- 《统计学习方法》例11.4 ----------
    def t1(y0, y1, x, i):
        """
        条件1: y0 in {0}          → 前一个状态必须是 0
        条件2: y1 in {1}          → 当前状态必须是 1
        条件3: x 的首位是 0        → 观测序列 x[0] == 0
        条件4: i in {1, 2}        → 只在位置 1 或位置 2 考察
         x in {(0,1,0), (0,1,1), (0,0,0), (0,0,1)} 这四个元组有一个共同点——首位都是 0，后两位任意。所以条件 3 本质上就是 x[0] == 0。
         "当观测序列的第一个元素是 0，且我们在考察第 2 或第 3 个位置时，如果前一个标签是 0、当前标签变成了 1，这个特征就被激活。"
        这是一个转移特征（tk），需要用到前一个状态 y0 = y_{i-1}。当 i = 0 时，y[-1] 没有意义（没有"位置 -1"的状态），
        所以转移特征只在 i >= 1 时才有定义。这里进一步限定只在 i = 1 或 i = 2 处生效。
        设计者认为："当观测序列以 0 开头时，从状态 0 转移到状态 1 是一个值得关注的模式"，所以把它定义为一个特征。权重 w1=1>0
 表示这个模式是受鼓励的。
        return 0,1
        """
        return int(y0 in {0} and y1 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1)} and i in {1, 2})


    def t2(y0, y1, x, i):
        return int(y0 in {0} and y1 in {0} and x in {(1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)} and i in {1})


    def t3(y0, y1, x, i):
        return int(y0 in {1} and y1 in {0, 1} and x in {(0, 0, 0), (1, 1, 1)} and i in {2})


    def t4(y0, y1, x, i):
        return int(y0 in {1} and y1 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1),
                                                     (1, 0, 0), (1, 0, 1)} and i in {2})


    def t5(y0, y1, x, i):
        return int(y0 in {0, 1} and y1 in {0} and x in {(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)} and i in {1, 2})


    def s1(y0, x, i):
        return int(y0 in {0} and x in {(0, 1, 1), (1, 1, 0), (1, 0, 1)} and i in {0, 1, 2})


    def s2(y0, x, i):
        return int(y0 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0),
                                       (1, 0, 1)} and i in {0})


    def s3(y0, x, i):
        return int(y0 in {0} and x in {(0, 1, 1), (1, 1, 0), (1, 0, 1)} and i in {0, 1})


    def s4(y0, x, i):
        return int(y0 in {1} and x in {(1, 0, 1), (0, 1, 0)} and i in {0, 2})


    w1 = [1, 0.6, 1.2, 0.2, 1.4]
    t = [t1, t2, t3, t4, t5]
    w2 = [1, 0.2, 0.8, 0.5]
    s = [s1, s2, s3, s4]

    X, Y = make_hidden_sequence(
        w1, t, w2, s,
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])



    for row in zip(X, Y):
        print(row)
