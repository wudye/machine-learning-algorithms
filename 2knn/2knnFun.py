from collections import Counter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1 -x2)))
    #return np.linalg.norm(x1 - x2)

def main(X_train, y_train, iris_classes, iris_point, k=5):  
    """
    test doc
    >>> addd= 12 + 1
    >>> addd
    13
    """
    distance = []
    ds = []
    lst = []
    for j in range(X_train.shape[0]):
        dist = euclidean_distance(X_train[j], iris_point)
        label = y_train[j]
        ds.append(dist)
        lst.append(label)
        distance.append((dist, label))
    sort_index = np.argsort(ds)[:k]
    cla = []
    for i in range(len(sort_index)):
        lb = lst[sort_index[i]]
        cla.append(lb)
    print(cla)
    # [(np.int64(2), 3)]
    res = Counter(cla).most_common(1)[0][0]
    return iris_classes[res]



if __name__ == "__main__":
    import doctest
    
    doctest.testmod()
    iris = load_iris()
    X = np.array(iris["data"])
    y = np.array(iris["target"])
    iris_classes = iris["target_names"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
    iris_point = np.array([4.4, 3.1, 1.3, 1.4])
    
    classifier = main(X_train, y_train, iris_classes, iris_point, k=5)
    print(classifier)
        