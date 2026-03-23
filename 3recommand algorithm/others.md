算法	改进点
FP-Growth	不生成候选集，只需 2 次扫描
PCY	第一遍扫描时用哈希过滤低频项
Sampling	采样数据预估计，减少扫描
Partition	分区处理，减少内存压力


排名	算法	流行度	原因
🥇	FP-Growth	⭐⭐⭐⭐⭐	效果最显著，被广泛实现
🥈	Apriori	⭐⭐⭐⭐	经典算法，教学首选
🥉	PCY	⭐⭐⭐	特定场景优化
4	Sampling	⭐⭐	精度有损失
5	Partition	⭐⭐	实现复杂度高