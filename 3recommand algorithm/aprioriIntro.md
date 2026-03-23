
TID     Items
1       Bread, Cokde, Milk
2       Beer, Bread
3       Beer, Coke, Diapper, Milk
4       Beer, Bread, Diapper, Milk
5       Coke, Diapper, Milk

Support(X, Y) = P(X, Y) = num(xy) / num(all samples)
Confidence(XY) = p(X|Y) = p(xy) / p(y)

{milk, diapper} , {bear}
support {milk, diapper,  bear} = 3, 4 => 2 / 5
cinfindence p(bear | {milk, diapper}) = p(3, 4 | 3, 4, 5) => 2 / 3

Interest = 2/3 - 2/5 = 4 / 15

Apriori algorithm:
Apriori定律1：如果一个集合是频繁项集，则它的所有子集都是频繁项集。 
Apriori定律2：如果一个集合不是频繁项集，则它的所有超集都不是频繁项集。
input -> Dataset ->D , support threshold aa
output -> most frequency dataset k
1）扫描整个数据集，得到所有出现过的数据，作为候选频繁1项集。

2）挖掘频繁k项集

a) 扫描数据计算候选频繁k项集的支持度

b) 去除候选频繁k项集中支持度低于阈值的数据集,得到频繁k项集。如果得到的频繁k项集为空，则直接返回频繁k-1项集的集合作为算法结果，算法结束。如果得到的频繁k项集只有一项，则直接返回频繁k项集的集合作为算法结果，算法结束。

c) 基于频繁k项集，连接生成候选频繁k+1项集。

3） 令k=k+1，转入步骤2。


data = [
    ['Bread', 'Coke', 'Milk'],              # TID 1
    ['Beer', 'Bread'],                       # TID 2
    ['Beer', 'Coke', 'Diaper', 'Milk'],      # TID 3
    ['Beer', 'Bread', 'Diaper', 'Milk'],     # TID 4
    ['Coke', 'Diaper', 'Milk']               # TID 5
]
min_support = 2  # 最小支持度阈值

第一轮：找频繁 1-项集
itemset = [['A'], ['B'], ['C'], ['A', 'B']]  # 候选项集
counts = [0, 0, 0, 0]                         # 初始化计数

逐交易计数：

交易	   检查过程	                                                                    counts 更新
TID 1: ['Bread','Coke','Milk']	Beer❌, Bread✅, Coke✅, Diaper❌, Milk✅	        [0,1,1,0,1]
TID 2: ['Beer','Bread']	Beer✅, Bread✅, Coke❌, Diaper❌, Milk❌	                [1,2,1,0,1]
TID 3: ['Beer','Coke','Diaper','Milk']	Beer✅, Bread❌, Coke✅, Diaper✅, Milk✅	[2,2,2,1,2]
TID 4: ['Beer','Bread','Diaper','Milk']	Beer✅, Bread✅, Coke❌, Diaper✅, Milk✅	[3,3,2,2,3]
TID 5: ['Coke','Diaper','Milk']	Beer❌, Bread❌, Coke✅, Diaper✅, Milk✅	        [3,3,3,3,4]

候选 1-项集：{Beer}, {Bread}, {Coke}, {Diaper}, {Milk}
项集	出现交易	计数	≥ 2?
{Beer}	2, 3, 4	    3	✅
{Bread}	1, 2, 4	    3	✅
{Coke}	1, 3, 5	    3	✅
{Diaper}	3, 4, 5	3	✅
{Milk}	1, 3, 4, 5	4	✅

频繁 1-项集：全部保留
itemset = [['Beer'], ['Bread'], ['Coke'], ['Diaper'], ['Milk']]  # 全部保留

第二轮：生成并过滤 2-项集
itemset = [['Beer','Bread'], ['Beer','Coke'], ['Beer','Diaper'], ['Beer','Milk'],
           ['Bread','Coke'], ['Bread','Diaper'], ['Bread','Milk'],
           ['Coke','Diaper'], ['Coke','Milk'], ['Diaper','Milk']]
counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

交易	检查结果	                                                                                        counts 更新
TID 1: ['Bread','Coke','Milk']	BB❌, BC❌, BD❌, BM❌, BrC❌, BrD❌, BrM✅, CD❌, CM✅, DM✅	        [0,0,0,0,0,0,1,0,1,1]
TID 2: ['Beer','Bread']	BB✅, BC❌, BD❌, BM❌, BrC❌, BrD❌, BrM❌, CD❌, CM❌, DM❌	                [1,0,0,0,0,0,1,0,1,1]
TID 3: ['Beer','Coke','Diaper','Milk']	BB❌, BC✅, BD✅, BM✅, BrC❌, BrD❌, BrM❌, CD✅, CM✅, DM✅	[1,1,1,1,0,0,1,1,2,2]
TID 4: ['Beer','Bread','Diaper','Milk']	BB✅, BC❌, BD✅, BM✅, BrC❌, BrD✅, BrM✅, CD❌, CM❌, DM✅	[2,1,2,2,0,1,2,1,2,3]
TID 5: ['Coke','Diaper','Milk']	BB❌, BC❌, BD❌, BM❌, BrC❌, BrD❌, BrM❌, CD✅, CM✅, DM✅	        [2,1,2,2,0,1,2,2,3,4]

候选 2-项集（两两组合）：
项集	出现交易	    计数	≥ 2?
{Beer, Bread}	2, 4	2	✅
{Beer, Coke}	3	    1	❌
{Beer, Diaper}	3, 4	2	✅
{Beer, Milk}	3, 4	2	✅
{Bread, Coke}	1	    1	❌
{Bread, Diaper}	4	    1	❌
{Bread, Milk}	1, 4	2	✅
{Coke, Diaper}	3, 5	2	✅
{Coke, Milk}	1, 3, 5	3	✅
{Diaper, Milk}	3, 4, 5	3	✅

频繁 2-项集：{Beer,Bread}, {Beer,Diaper}, {Beer,Milk}, {Bread,Milk}, {Coke,Diaper}, {Coke,Milk}, {Diaper,Milk}
# counts = [2, 2, 2, 2, 2, 3, 4]


第三轮：生成并过滤 3-项集
候选 3-项集（基于频繁 2-项集组合）：
itemset = [['Beer','Bread','Milk'], ['Beer','Diaper','Milk'], ['Coke','Diaper','Milk']]
counts = [0, 0, 0]

项集	                出现交易    计数	≥ 2?
{Beer, Bread, Milk}	    4	        1	❌
{Beer, Diaper, Milk}	3, 4	    2	✅
{Coke, Diaper, Milk}	3, 5	    2	✅
频繁 3-项集：{Beer,Diaper,Milk}, {Coke,Diaper,Milk}
# counts = [2, 3]


第四轮：生成 4-项集
无法组合生成候选（没有共同的子集），算法结束。
最终结果
频繁项集	                    支持度计数
{Beer}	                        3
{Bread}	                        3
{Coke}	                        3
{Diaper}	                    3
{Milk}	                        4
{Beer, Bread}	                2
{Beer, Diaper}	                2
{Beer, Milk}	                2
{Bread, Milk}	                2
{Coke, Diaper}	                2
{Coke, Milk}	                3
{Diaper, Milk}	                3
{Beer, Diaper, Milk}	        2
{Coke, Diaper, Milk}	        2
关联规则示例：

买 {Beer, Diaper} 的人 → 买 Milk（置信度 100%）
买 {Coke, Diaper} 的人 → 买 Milk（置信度 100%）