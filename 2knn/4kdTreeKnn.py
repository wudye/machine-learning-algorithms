import heapq
from collections import Counter
import numpy as np


def euclidean_distance(x1, x2):
    return np.linalg.norm(x1,  x2)


class Node:
    __slots__ = "element", "axis", "left", "right"

    def __init__(self, element, axis=0, left=None, right=None):
        self.element = element
        self.axis = axis
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.element < other.element


class KDTree:
    def __init__(self, data, distance_func):
        self._data = list(data)
        self._size = len(data)
        self._distance_func = distance_func
        if self._size > 0:
            self._dimension = len(data[0])
            self._root = self._build_kd_tree(list(data), depth=0)
        else:
            self._dimension = 0
            self._root = None
    
    def _build_kd_tree(self,  data,  depth):
        if not data



class KDTree1:
    class _Node:
        """kd树的轻量级结点"""
        __slots__ = "element", "axis", "left", "right"

        def __init__(self, element, axis=0, left=None, right=None):
            self.element = element
            self.axis = axis
            self.left = left
            self.right = right

        def __lt__(self, other):
            return self.element < other.element

    def __init__(self, data, distance_func):
        """构造平衡kd树实例"""
        self._data = list(data)  # ★ 保存原始数据
        self._size = len(data)
        self._distance_func = distance_func
        if self._size > 0:
            self._dimension = len(data[0])
            self._root = self._build_kd_tree(list(data), depth=0)  # ★ 传副本
        else:
            self._dimension = 0
            self._root = None

    def _build_kd_tree(self, data, depth):
        """构造平衡kd树"""
        if not data:
            return None

        select_axis = depth % self._dimension
        data = sorted(data, key=lambda x: x[select_axis])  # ★ 用 sorted，不修改原列表
        median_index = len(data) // 2

        node = self._Node(data[median_index], axis=select_axis)
        node.left = self._build_kd_tree(data[:median_index], depth + 1)
        node.right = self._build_kd_tree(data[median_index + 1:], depth + 1)
        return node

    def search_knn(self, x, k):
        """返回距离x最近的k个点"""
        res = []
        self._search_knn(res, self._root, x, k)
        return [(node.element, -distance) for distance, node in sorted(res, key=lambda xx: -xx[0])]

    def _search_knn(self, res, node, x, k):
        if node is None:
            return

        node_distance = self._distance_func(node.element, x)
        node_distance_axis = abs(node.element[node.axis] - x[node.axis])

        # 处理当前结点
        if len(res) < k:
            heapq.heappush(res, (-node_distance, node))
        elif node_distance < (-res[0][0]):
            heapq.heappushpop(res, (-node_distance, node))

        # 处理目标点所在的子结点
        if x[node.axis] <= node.element[node.axis]:
            self._search_knn(res, node.left, x, k)
        else:
            self._search_knn(res, node.right, x, k)

        # 处理目标点不在的子结点（剪枝）
        if len(res) < k or node_distance_axis < (-res[0][0]):
            if x[node.axis] <= node.element[node.axis]:
                self._search_knn(res, node.right, x, k)
            else:
                self._search_knn(res, node.left, x, k)

    def print_tree(self, node=None, prefix="", is_last=True):
        """打印KDTree的树形结构"""
        if node is None:
            node = self._root
            print("KDTree Structure:")
            print(f"总节点数: {self._size}, 维度: {self._dimension}")
            print("─" * 40)
        
        if node is None:
            return
        
        # 当前节点信息
        connector = "└── " if is_last else "├── "
        axis_name = ["X", "Y", "Z", "W"][node.axis] if node.axis < 4 else f"轴{node.axis}"
        print(f"{prefix}{connector}[{axis_name}] {node.element}")
        
        # 准备子节点的前缀
        extension = "    " if is_last else "│   "
        child_prefix = prefix + extension
        
        # 统计子节点数量
        children = []
        if node.left:
            children.append(("L", node.left))
        if node.right:
            children.append(("R", node.right))
        
        # 递归打印子节点
        for i, (label, child) in enumerate(children):
            is_last_child = (i == len(children) - 1)
            sub_connector = "└── " if is_last_child else "├── "
            print(f"{child_prefix}{sub_connector}{label}:")
            self.print_tree(child, child_prefix + extension, is_last_child)

    def display(self, node=None, depth=0):
        """简单缩进打印"""
        if node is None:
            node = self._root
        if node is None:
            return
        
        indent = "  " * depth
        print(f"{indent}Depth {depth}, Axis {node.axis}: {node.element}")
        
        if node.left:
            print(f"{indent}  ↙ Left:")
            self.display(node.left, depth + 1)
        if node.right:
            print(f"{indent}  ↘ Right:")
            self.display(node.right, depth + 1)

    from collections import deque

    def print_level_order(self):
        """按层级打印（从上到下，从左到右）"""
        if self._root is None:
            print("空树")
            return
        
        queue = deque([(self._root, 0)])
        current_level = 0
        print(f"Level {current_level}:")
        
        while queue:
            node, level = queue.popleft()
            
            if level > current_level:
                current_level = level
                print(f"\nLevel {current_level}:")
            
            axis_name = ["X", "Y", "Z"][node.axis] if node.axis < 3 else f"轴{node.axis}"
            print(f"  [{axis_name}] {node.element}", end="")
            
            if node.left or node.right:
                children = []
                if node.left:
                    children.append("L")
                    queue.append((node.left, level + 1))
                if node.right:
                    children.append("R")
                    queue.append((node.right, level + 1))
                print(f" (Children: {', '.join(children)})")
            else:
                print(" (Leaf)")

# ========== 新增的 KNN 分类类 ==========
class KDTreeKNN:
    """基于KDTree的K近邻分类器"""
    
    def __init__(self, X_train, y_train, k=5):
        """
        :param X_train: 训练数据特征，如 [[3,3], [4,3], [1,1]]
        :param y_train: 训练数据标签，如 [1, 1, -1]
        :param k: 邻居数量
        """
        self.k = k
        self.y_train = y_train
        # 构建KDTree，只传入特征数据
        self.kdtree = KDTree(X_train, euclidean_distance)
    
    def predict(self, x):
        """
        预测单个样本的类别
        :param x: 待预测样本，如 [3, 4]
        :return: 预测的类别标签
        """
        # 搜索k个最近邻（返回的是训练集中的点）
        neighbors = self.kdtree.search_knn(x, self.k)
        
        # 统计k个邻居的标签
        count = Counter()
        for point, distance in neighbors:
            # 找到这个点在训练集中的索引，获取标签
            for i, train_point in enumerate(self.kdtree._data if hasattr(self.kdtree, '_data') else []):
                if list(point) == list(train_point):
                    count[self.y_train[i]] += 1
                    break
        
        return count.most_common(1)[0][0]



# ========== 测试代码 ==========
if __name__ == "__main__":
    X_train = [(3, 3), (4, 3), (1, 1), (2, 2), (5, 5)]
    
    # 创建KDTree
    kdtree = KDTree(X_train, euclidean_distance)
    
    # 打印树结构
    kdtree.print_tree()
    # 训练数据：3个样本，2维特征
    X_train = [(3, 3), (4, 3), (1, 1)]
    y_train = [1, 1, -1]
    
    # 创建KNN分类器，k=2
    knn = KDTreeKNN(X_train, y_train, k=2)

    
    # 预测点 (3, 4)
    result = knn.predict((3, 4))
    print(f"点 (3,4) 的预测类别: {result}")  # 应该输出 1
