# Perceptron Dual Form
# hyperplane sum(alpha_i * yi * xi) + b = 0
# alpha_i, eta(0, 1) -> learning rate
# a sample (xi, yi)
# yi(sum(alpha_i * yi * xi) + b ) > 0 , right classification
# yi(sum(alpha_i * yi * xi) + b ) <= 0 , wrong classification
# gram matrix Gij = xi * xj
"""
x = [(3, 3), (4, 3), (1, 1)]
Gram matrix G:
G11: x1 * x1 = 3*3 + 3*3 = 18 (x11 * x11 + x12 * x12)
G12: x1 * x2 = 3*4 + 3*3 = 21 (x11 * x21 + x12 * x22)
G13: x1 * x3 = 3*1 + 3*1 = 6 (x11 * x31 + x12 * x32)
G21: x2 * x1 = 4*3 + 3*3 = 21 (x21 * x11 + x22 * x12)
G22: x2 * x2 = 4*4 + 3*3 = 25 (x21 * x21 + x22 * x22)
G23: x2 * x3 = 4*1 + 3*1 = 7 (x21 * x31 + x22 * x32)
G31: x3 * x1 = 1*3 + 1*3 = 6 (x31 * x11 + x32 * x12)
G32: x3 * x2 = 1*4 + 1*3 = 7 (x31 * x21 + x32 * x22)
G33: x3 * x3 = 1*1 + 1*1 = 2 (x31 * x31 + x32 * x32)
G = [[18, 21, 6],
     [21, 25, 7],
     [6, 7, 2]]
"""
# update by loss function L(alpha, b) = - sum(alpha_i * yi(sum(alpha_j * yj * xj) + b))
# alpha_i = alpha_i + eta * yi
# b = b + eta * yi
# iterate from 0 to number_sample-1 until all samples are correctly classified


def gram_matrix(x):
    number_sample = len(x)
    number_feature = len(x[0])
    gram = [[0] * number_sample for _ in range(number_sample)]

    for i in range(number_sample):
        for j in range(i, number_sample):
            gram[i][j] = gram[j][i] = sum(x[i][k] * x[j][k] for k in range(number_feature))
    
    return gram



    

def dual_perceptron(x, y, eta):
    number_sample = len(x)

    alpha = [0] * number_sample
    b0 = 0
    gram = gram_matrix(x)

    while True:
        for i in range(number_sample):
            yi = y[i]
            val = 0

            for j in range(number_sample):
                xj = x[j]
                yj = y[j]
                val += alpha[j] * yj * gram[i][j]
            if yi * (val + b0) <= 0:
                alpha[i] += eta 
                b0 += eta * yi
                break
        else:
            return alpha, b0


if __name__ == "__main__":
    x = [(3, 3), (4, 3), (1, 1)]  
    y = [1, 1, -1]
    eta = 0.5
    w, b = dual_perceptron(x, y, eta)
    print(w, b)