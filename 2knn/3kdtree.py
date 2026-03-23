from collections import Counter
from sklearn.neighbors import KDTree

class KDTreeKNN:
    
    def __init__(self, x, y, k, metric="euclidean"):
        self.x, self.y, self.k = x, y, k
        self.kdtree = KDTree(self.x, metric=metric)
    
    def count(self, x):
        index = self.kdtree.query([x], self.k, return_distance=False)
        count = Counter()
        for i in index[0]:
            count[self.y[i]] += 1
        return count.most_common(1)[0][0]
    

if __name__ == "__main__":
    dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]] 
    knn = KDTreeKNN(dataset[0], dataset[1], k=2)
    print(knn.count((3, 4)))    