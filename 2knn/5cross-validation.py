from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def cross_validation_example_knn(x, y):
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=42)
    x11, x12, y11, y12 = train_test_split(x1, y1, test_size=0.2, random_state=42)
    best_k, best_score = 0, 0
    for k in range(1, 101):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x11, y11)
        score = knn.score(x12, y12)
        if score > best_score:
            best_k, best_score = k,  score
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x2, y2)
    return best_k, best_knn.score(x2, y2)

if __name__ == "__main__":
    X, Y = make_blobs(n_samples = 1000,
                      n_features=10,
                      centers=5,
                      cluster_std=5000,
                      center_box=(-10000, 10000),
                      random_state=42)
    k, score = cross_validation_example_knn(X, Y)
    print("optimise k", k)
    print("best score", score)