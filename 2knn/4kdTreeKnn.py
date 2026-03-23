import heapq
from collections import Counter
from sys import prefix
import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

class KDTree:
    class _Node:
        # use slots to set allowed paramters , save memory
        __slots__ = "element", "axis", "left", "right"

        def __init__(self, element, axis=0, left=None, right=None):
            self.element = element
            self.axis = axis
            self.left = left
            self.right = right

        def __lt__(self, other):
            return self.element < other.element

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

    def _build_kd_tree(self, data, depth):
        if not data:
            return None

        select_axis = depth % self._dimension
        data = sorted(data, key=lambda x: x[select_axis])  # use sorted no modify original
        median_index = len(data) // 2

        node = self._Node(data[median_index], axis=select_axis)
        node.left = self._build_kd_tree(data[:median_index], depth + 1)
        node.right = self._build_kd_tree(data[median_index + 1:], depth + 1)
        return node

    def search_knn(self, x, k):
        
        res = []
        self._search_knn(res, self._root, x, k)
        return [(node.element, -distance) for distance, node in sorted(res, key=lambda xx: -xx[0])]

    def _search_knn(self, res, node, x, k):
        if node is None:
            return

        node_distance = self._distance_func(node.element, x)
        node_distance_axis = abs(node.element[node.axis] - x[node.axis])

        # current node handle
        if len(res) < k:
            heapq.heappush(res, (-node_distance, node))
        elif node_distance < (-res[0][0]):
            heapq.heappushpop(res, (-node_distance, node))
        # child node
        if x[node.axis] <= node.element[node.axis]:
            self._search_knn(res, node.left, x, k)
        else:
            self._search_knn(res, node.right, x, k)

        # prune
        if len(res) < k or node_distance_axis < (-res[0][0]):
            if x[node.axis] <= node.element[node.axis]:
                self._search_knn(res, node.right, x, k)
            else:
                self._search_knn(res, node.left, x, k)

    def print_tree(self, node=None, prefix="", is_last=True):
        if node is None:
            node = self._root
            print("KDTree Structure:")
            print(f"nodes: {self._size}, dimension: {self._dimension}")
            print("─" * 40)
        
        if node is None:
            return
        
        connector = "└── " if is_last else "├── "
        axis_name = ["X", "Y", "Z", "W"][node.axis] if node.axis < 4 else f"轴{node.axis}"
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
            print(f"{child_prefix}{sub_connector}{label}:")
            self.print_tree(child, child_prefix + extension, is_last_child)

    def display(self, node=None, depth=0):
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
        if self._root is None:
            print("empty")
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

class KDTreeKNN:
    
    def __init__(self, X_train, y_train, k=5):
        self.k = k
        self.y_train = y_train
        self.kdtree = KDTree(X_train, euclidean_distance)
    
    def predict(self, x):
        neighbors = self.kdtree.search_knn(x, self.k)
        
        count = Counter()
        for point, distance in neighbors:
            for i, train_point in enumerate(self.kdtree._data if hasattr(self.kdtree, '_data') else []):
                if list(point) == list(train_point):
                    count[self.y_train[i]] += 1
                    break
        
        return count.most_common(1)[0][0]



if __name__ == "__main__":
    X_train = [(3, 3), (4, 3), (1, 1), (2, 2), (5, 5)]
    
    kdtree = KDTree(X_train, euclidean_distance)
    
    kdtree.print_tree()
    X_train = [(3, 3), (4, 3), (1, 1)]
    y_train = [1, 1, -1]
    
    knn = KDTreeKNN(X_train, y_train, k=2)

    result = knn.predict((3, 4))
    print(f" (3,4) belongs : {result}")  # 1
