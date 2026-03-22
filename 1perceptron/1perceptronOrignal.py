# hyperplane wx + b = 0
# w0, eta(0, 1) -> learning rate
# a sample (xi, yi) 
# yi(wi + b ) > 0 , right classification
# yi(wi + b ) <= 0 , wrong classification
# update by loss function L(w, b) = - sum(yi(wi + b))
# w = w + eta * yi * xi
# b = b + eta * yi
# iterate from 0 to number_sample-1 until all samples are correctly classified



def perceptron(x, y, eta):
    number_sample = len(x)
    number_feature = len(x[0])

    w0 = [0] * number_feature
    b0 = 0
    count = 0
    while True:
        print("count: ", count)
        for i in range(number_sample):
            xi, yi = x[i], y[i]
            print(xi, yi, i)
            if yi * (sum(w0[j] * xi[j] for j in range (number_feature)) + b0) <= 0:
                w1 = [w0[j] + eta * yi * xi[j] for j in range (number_feature)]
                b1 = b0 + eta * yi
                print("update: ", w1, b1)
                w0 = w1
                b0 = b1
                count += 1
                break
        else:
            return w0, b0
    
if __name__ == "__main__":
    x = [(3, 3), (4, 3), (1, 1)]  
    y = [1, 1, -1]
    eta = 0.5
    w, b = perceptron(x, y, eta)
    print(w, b)
