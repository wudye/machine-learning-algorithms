1. 内存效率：生成器 vs 列表
    2的方式 — 列表存所有（假设 n=100万）
    distance = []
    for j in range(X_train.shape[0]):  # 100万次循环
        distance.append((dist, label))  # 存100万个元组
    # 内存占用：~几十MB

    # 类的方式 — 生成器惰性计算
    distance = (
        (self._euclidean_distance(...), data_point[1]) 
        for data_point in self.data)  # 不存储，只定义计算规则
    # 内存占用：几乎为0
2. 排序效率：全排 vs Top-K
    # 2的方式 — 全部排序（浪费！）
    sort_index = np.argsort(ds)[:k]  # 排序100万个，只要5个

    # 类的方式 — 只找最小的k个
    votes = (i[1] for i in nsmallest(k, distance))  # 堆算法，只维护k个

3. heapq.nsmallest 的原理
    # nsmallest 内部使用最小堆
    # 只维护大小为 k 的堆，不需要存储全部数据

    # 模拟过程（k=3）：
    数据流: [9, 2, 7, 1, 5, 3...]
    堆状态: [9] → [2,9] → [2,7,9] → [1,2,7] → [1,2,5] → [1,2,3]
            只保留3个最小的，新来的大的直接丢弃

惰性计算 + Top-K 堆算法，这是处理大数据的标准做法。

生成器是一种惰性计算的工具，它不会立即执行，而是在你需要时才逐个产生值。

    # 列表推导式 [] —— 立刻计算所有，存内存
    my_list = [x * 2 for x in range(1000000)]  # 占用内存：~8MB

    # 生成器表达式 () —— 按需计算，不存内存
    my_gen = (x * 2 for x in range(1000000))   # 占用内存：~几十字节

    生成器的特点
特性	列表	生成器
创建	[x for x in range(n)]	(x for x in range(n))
内存	存储所有元素	只存生成规则
访问	可索引 list[0]	不可索引
遍历次数	无限次	只能遍历一次
速度	创建慢，访问快	创建快，访问时计算

惰性计算 (策略)
    ├── 生成器 (generator)
    ├── 迭代器 (iterator)
    ├── map/filter
    └── 惰性求值的语言特性

Top-K 问题 (找最小/最大的K个)
    ├── nsmallest (堆实现，O(n log k))
    ├── nlargest
    ├── 全排序后切片 (O(n log n))
    └── np.partition (快速选择算法)
