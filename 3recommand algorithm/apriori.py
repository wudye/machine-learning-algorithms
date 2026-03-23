"""
Apriori Algorithm is a Association rule mining technique, also known as market basket
analysis, aims to discover interesting relationships or associations among a set of
items in a transactional or relational database.

For example, Apriori Algorithm states: "If a customer buys item A and item B, then they
are likely to buy item C."  This rule suggests a relationship between items A, B, and C,
indicating that customers who purchased A and B are more likely to also purchase item C.

WIKI: https://en.wikipedia.org/wiki/Apriori_algorithm
Examples: https://www.kaggle.com/code/earthian/apriori-association-rules-mining
"""

from collections import Counter
from itertools import combinations


def load_data() -> list[list[str]]:
    """
    Returns a sample transaction dataset.

    >>> load_data()
    [['milk'], ['milk', 'butter'], ['milk', 'bread'], ['milk', 'bread', 'chips']]
    """
    # return [["milk"], ["milk", "butter"], ["milk", "bread"], ["milk", "bread", "chips"]]
    data = [
    ['Bread', 'Coke', 'Milk'],              # TID 1
    ['Beer', 'Bread'],                       # TID 2
    ['Beer', 'Coke', 'Diaper', 'Milk'],      # TID 3
    ['Beer', 'Bread', 'Diaper', 'Milk'],     # TID 4
    ['Coke', 'Diaper', 'Milk']               # TID 5
    ]
    return data

def generate_candidates(itemset: list, k: int) -> list:
    """
    从 (k-1)-项集生成 k-项集候选。
    
    :param itemset: 频繁 (k-1)-项集列表，每个项集是排序列表
    :param k: 目标项集大小
    :return: k-项集候选列表
    itemset = [
    ['Beer', 'Bread'],      # i=0
    ['Beer', 'Diaper'],     # i=1
    ['Beer', 'Milk'],       # i=2
    ['Bread', 'Milk'],      # i=3
    ['Coke', 'Diaper'],     # i=4
    ['Coke', 'Milk'],       # i=5
    ['Diaper', 'Milk']      # i=6
    ]
    -> 
    candidates = [
    ['Beer', 'Bread', 'Diaper'],
    ['Beer', 'Bread', 'Milk'],
    ['Beer', 'Diaper', 'Milk'],
    ['Coke', 'Diaper', 'Milk']
    ]

    """
    candidates = []
    n = len(itemset)
    #print("oooo->",itemset)
    for i in range(n):
        for j in range(i + 1, n):
            # 前k-2个元素相同才能合并（避免重复） [:-1] 表示"取除最后一个元素外的所有元素"：
            # itemset[i][:-1]	第 i 个项集的前 k-2 个元素
            # itemset[j][:-1]	第 j 个项集的前 k-2 个元素
            # itemset[j][-1] get the last one
            #print("aaa-> ",itemset[i][:-1], itemset[j][-1] )
            if itemset[i][:-1] == itemset[j][:-1]:
                candidate = sorted(itemset[i] + [itemset[j][-1]])
                if candidate not in candidates:
                    candidates.append(candidate)
    #print("aaaaaaa->", candidates )
    return candidates


def prune(itemset: list, candidates: list) -> list:
    """
    剪枝：候选的所有 (k-1) 子集必须是频繁的。
    """
    print("2222->", candidates)
    print("333->",   itemset)
    frequent_set = set(tuple(item) for item in itemset)
    print("ddd.>", frequent_set)
    pruned = []
    for candidate in candidates:
        # 检查所有 (k-1) 子集是否都在频繁项集中
        is_valid = True
        for i in range(len(candidate)):
            subset = tuple(candidate[:i] + candidate[i+1:])
            print("444->", subset)
            if subset not in frequent_set:
                is_valid = False
                break
        if is_valid:
            pruned.append(candidate)
    return pruned


def apriori(data: list[list[str]], min_support: int) -> list[tuple[list[str], int]]:
    """
    Returns a list of frequent itemsets and their support counts.

    >>> data = [['A', 'B', 'C'], ['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C']]
    >>> apriori(data, 2)
    [(['A', 'B'], 1), (['A', 'C'], 2), (['B', 'C'], 2)]

    >>> data = [['1', '2', '3'], ['1', '2'], ['1', '3'], ['1', '4'], ['2', '3']]
    >>> apriori(data, 3)
    []
    """
    # 初始化：获取所有唯一商品作为 1-项集候选
    unique_items = set()
    for transaction in data:
        for item in transaction:
            unique_items.add(item)
    itemset = [[item] for item in sorted(unique_items)]  # 排序保证一致性

    frequent_itemsets = []
    k = 1  

    while itemset:
        print(f"\n=== Round {k}: generating {k}-itemsets ===")
       # print("Candidates:", itemset)
        
        # 统计每个候选项集的支持度
        counts = [0] * len(itemset)
        for transaction in data:
            for j, candidate in enumerate(itemset):
                if all(item in transaction for item in candidate):
                    counts[j] += 1
        # print("Counts:", counts)

        # 过滤：保留满足最小支持度的项集
        """
        frequent = []
        for i in range(len(itemset)):           # 遍历每个候选项集的索引
            if counts[i] >= min_support:        # 如果支持度 ≥ 阈值
                frequent.append((sorted(itemset[i]), counts[i]))  # 添加 (排序后的项集, 计数)

        """
        frequent = [(sorted(itemset[i]), counts[i]) 
                    for i in range(len(itemset)) if counts[i] >= min_support]
        # print("Frequent:", [f[0] for f in frequent])
        
        frequent_itemsets.extend(frequent)

        # 生成下一轮候选
        k += 1
        """
        frequent = [
            (['Beer', 'Bread'], 2),
            (['Beer', 'Diaper'], 2),
            (['Beer', 'Milk'], 2),
            (['Bread', 'Milk'], 2),
            (['Coke', 'Diaper'], 2),
            (['Coke', 'Milk'], 3),
            (['Diaper', 'Milk'], 3)
        ]
        prev_frequent = []
        for item, count in frequent:    # 解包元组 (item, count)
            prev_frequent.append(item)  # 只保留 item

        """
        prev_frequent = [item for item, count in frequent]  # 只保留项集，不含计数
        candidates = generate_candidates(prev_frequent, k)
        itemset = prune(prev_frequent, candidates)

    return frequent_itemsets


if __name__ == "__main__":
    """
    Apriori algorithm for finding frequent itemsets.

    Args:       
        data: A list of transactions, where each transaction is a list of items.
        min_support: The minimum support threshold for frequent itemsets.

    Returns:
        A list of frequent itemsets along with their support counts.
    """
    import doctest

    #doctest.testmod()


    # user-defined threshold or minimum support level
    frequent_itemsets = apriori(data=load_data(), min_support=2)
    print("\n".join(f"{itemset}: {support}" for itemset, support in frequent_itemsets))