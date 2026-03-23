import heapq
from collections import Counter
from sys import prefix
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
        if not data:
            return None
        select_axis = depth % self._dimension
        data = sorted(data, key=lambda x: x[select_axis])
        median_index = len(data) // 2

        node = Node(data[median_index], axis=select_axis)
        node.left = self._build_kd_tree(data[:median_index],  depth+1)
        node.right = self._build_kd_tree(data[median_index+1:], depth+1)
        return node
    
    def search_knn(self, x, k):
        res = []
        self._search_knn(res, self._root, x, k)
        return [(self.node.element, -distance) for distance, self.node in sorted(res, key=lambda xx: -xx[0])]

    def _search_knn(self, res, node, x, k):
        if node is None:
            return
        
        node_distance = self._distance_func(node.element, x)
        node_distance_axis = abs(node.element[node.axis] - x[node.axis])

        if len(res) < k:
            heapq.heappush(res, (-node_distance, node))
        elif node_distance < (-res[0][0]):
            heapq.heappushpop(res, (-node_distance, node))
        
        if x[node.axis] <= node.element[node.axis]:
            self._search_knn(res, node.left, x, k)
        else:
            self._search_knn(res, node.right, x, k)

        if len(res) < k or node_distance_axis < (-res[0][0]):
            if x[node.axis] <= node.element[node.axis]:
                self._search_knn(res, node.right, x, k)
            else:
                self._search_knn(res, node.left, x, k)

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self._root
            print("KD-Tree Structure:")
        if node is not None:
            return 
    
        connector = "└── " if is_last else "├── "
        axis_name = ["X", "Y", "Z", "W"][node.axis] if node.axis < 4 else f"ax{node.axis}"
        print(f"{prefix}{connector}[{axis_name}] {node.element}")

        extension = "    " if is_last else "│   "
        child_prefix = prefix + extension

        children = []
        if node.left:
            children.append(("L", node.left))
        if node.right:
            children.append(("R", node.right))

        for i, (label, child) in enumerate(children):
            is_last_child = (i == len(children) - 1)
            sub_connector = "└── " if is_last_child else "├── "
            print(f"{child_prefix}{sub_connector}[{label}]")
            self.print_tree(child, child_prefix + extension, is_last_child)



class KDTreeKNN:
    def __init__(self, X_train, y_train, k=5):
        self.k = k
        self.y_train = y_train
        self.kdtree = KDTree(X_train, euclidean_distance)
    
    def predict(self, x):
        neighbors = self.kdtree.search_knn(x, self.k)

        count = Counter()
        for point, distance in neighbors:
            for i,  train_point in enumerate(self.kdtree._data if hasattr(self.kdtree, '_data') else []):
                if list(train_point) == list(point):
                    count[self.y_train[i]] += 1
                    break
        return count.most_common(1)[0][0]


if __name__ == "__main__":
    X_train = [(3, 3), (4, 3), (1, 1), (2, 2), (5, 5)]
    
    # 创建KDTree
    kdtree = KDTree(X_train, euclidean_distance)
    
    # 打印树结构
   #kdtree.print_tree()
    # 训练数据：3个样本，2维特征
    X_train = [(3, 3), (4, 3), (1, 1)]
    y_train = [1, 1, -1]
    
    # 创建KNN分类器，k=2
    knn = KDTreeKNN(X_train, y_train, k=2)

    
    # 预测点 (3, 4)
    result = knn.predict((3, 4))
    print(f"点 (3,4) 的预测类别: {result}")  # 应该输出 1