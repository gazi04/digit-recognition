from numpy import exp, maximum, ndarray

def relu(z):
    return maximum(0, z)

def sigmoid(z):
    return 1 / (1 + exp(-z))

def relu_derivative(z: ndarray):
    return (z>0).astype(int)

def softmax(z: ndarray):
    return exp(z) / sum(exp(z))
