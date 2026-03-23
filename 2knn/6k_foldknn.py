from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def k_cross_validation_example_knn(x, y):
    best_k, best_score = 0, 0
    for k in range(1, 101):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x, y, cv=10, scoring="accuracy")
        score = scores.mean()
        if score > best_score:
            best_k, best_score = k,  score
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x, y)
    return best_k, best_score

if __name__ == "__main__":
    X, Y = make_blobs(n_samples = 1000,
                      n_features=10,
                      centers=5,
                      cluster_std=5000,
                      center_box=(-10000, 10000),
                      random_state=42)
    k, score = k_cross_validation_example_knn(X, Y)
    print("optimise k", k)
    print("best score", score)