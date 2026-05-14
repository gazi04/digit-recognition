from numpy import arange, exp, maximum, ndarray, zeros, sum

def relu(z):
    return maximum(0, z)

def sigmoid(z):
    return 1 / (1 + exp(-z))

def relu_derivative(z: ndarray):
    return (z>0).astype(int)

def softmax(z: ndarray):
    return exp(z) / sum(exp(z), axis=0)

def one_hot(Y: ndarray) -> ndarray:
    one_hot_Y = zeros((Y.size, Y.max() + 1))
    one_hot_Y[arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
