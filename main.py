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

if __name__ == "__main__":
    main()
