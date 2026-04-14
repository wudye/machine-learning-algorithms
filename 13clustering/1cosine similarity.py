import numpy as np



def cosine(x):

    n = len(x[0])
    sqaure = [np.square(x[:, i]).sum() for i in range(n)]
    print(sqaure)

    D = np.identity(n)

    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = x[:, i], x[:, j]
            numerator = (xi * xj).sum()
            denominator = np.sqrt(sqaure[i] * sqaure[j])
            if denominator:
                D[i][j] = D[j][i] = numerator / denominator
            else:
                D[i][j] = D[j][i] = np.nan
    return D


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 5, 5],
                  [2, 0, 0, 0, 2]])


    print(cosine(X))

    # [[1.                nan 0.         0.         0.37139068]
    #  [       nan 1.                nan        nan        nan]
    #  [0.                nan 1.         1.         0.92847669]
    #  [0.                nan 1.         1.         0.92847669]
    #  [0.37139068        nan 0.92847669 0.92847669 1.        ]]