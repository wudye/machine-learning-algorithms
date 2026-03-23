from collections import Counter
from heapq import nsmallest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

class KNN:
    def __init__(self, train_data: np.ndarray[float],train_target: np.ndarray[int],
        class_labels: list[str]):

        
        self.data = zip(train_data, train_target)
        self.labels = class_labels

    @staticmethod
    def _euclidean_distance(a: np.ndarray[float], b:np.ndarray[float]) -> float:
        # np.sqrt(np.sum((a - b) ** 2))
        res =  float(np.linalg.norm(a-b))
        #print(a, b, res)
        # return float(np.linalg.norm(a-b))
        return res
    def classify(self, pred_point: np.ndarray[float], k:int=5) -> str:

        distance = (
            (self._euclidean_distance(data_point[0], pred_point), data_point[1]) 
                     for data_point in self.data)
        #print("dist->", distance)
        votes = (i[1] for i in nsmallest(k, distance))
        #for i in votes:
        #    print(i)
        res = Counter(votes).most_common(1)[0][0]
        #print(res)
        return self.labels[res]

if __name__ == "__main__":
    import doctest
    
    doctest.testmod()
    iris = load_iris()
    X = np.array(iris["data"])
    y = np.array(iris["target"])
    iris_classes = iris["target_names"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
    iris_point = np.array([4.4, 3.1, 1.3, 1.4])
    classifier = KNN(X_train, y_train, iris_classes)
    print(classifier.classify(iris_point, k=5))
        
   


        

