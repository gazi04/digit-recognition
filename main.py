from typing import Dict
from numpy.random import randn
from pandas import read_csv
from numpy import dot, log, ndarray, sum
from .activation_functions import one_hot, relu, sigmoid, softmax


def main():
    print("Hello from ann-from-scratch!")

def init_weights() -> Dict[str, ndarray]:
    weights1 = randn(10, 784)
    bias1 = randn(10, 1)
    weights2 = randn(10, 10)
    bias2 = randn(10, 1)
    return {
        "w1": weights1,
        "b1": bias1,
        "w2":weights2,
        "b2": bias2,
    }

def feed_forward(w1: ndarray, b1: ndarray, w2: ndarray, b2: ndarray, x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2

def calculate_loss(predictions: ndarray, results: ndarray):
    results = one_hot(results)
    return -(sum(results) * log(predictions))

def backward_propagation(z1: ndarray, a1: ndarray, a2: ndarray, w2: ndarray, x: ndarray, y: ndarray, m: int):
    dz2 = a2 - y # the error term of the output layer
    dw2 = (1/m) * dz2.dot(a1.T) # weighted gradient for layer 2 (output layer)
    db2 = (1/m) * sum(dz2, axis=1, keepdims=True) # bias gradient for layer 2 

    dz1 = w2.T.dot(dz2) * relu_derivative(z1)
    dw1 = (1/m) * dz1.dot(x.T)
    db1 = (1/m) * sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2

def update_parameters(w1, w2, b1, b2, dw1, db1, dw2, db2, alpha):
    w2 = w2 - (dw2*alpha)
    b2 = b2 - (db2*alpha)
    w1 = w1 - (dw1*alpha)
    b1 = b1 - (db1*alpha)

    return w1, w2, b1, b2
if __name__ == "__main__":
    main()
