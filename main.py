from numpy.random import randn
from pandas import read_csv
from numpy import  argmax, array, log, ndarray, random, sum
from activation_functions import one_hot, relu, relu_derivative, softmax
from matplotlib import pyplot as plt

def main():
    data = read_csv("digit-recognizer/train.csv")
    data = array(data)
    random.shuffle(data)
    data = data.T

    # --- TESTING DATA ---
    # Give me ALL rows (:), but only the first 10 columns (0:10)
    data_test = data[:, 0:10] 
    y_test = data_test[0]                 # Row 0 is the label
    testing_data = data_test[1:] / 255.0  # Rows 1-784 are the pixels
    
    # --- TRAINING DATA ---
    # Give me ALL rows (:), but only columns from 10 to the end (10:)
    data_train = data[:, 10:]
    y_train = data_train[0]                  # Row 0 is the label
    training_data = data_train[1:] / 255.0   # Rows 1-784 are the pixels

    w1, w2, b1, b2 = traning_loop(training_data, y_train, 0.1, 2000)
    
    for i in range(10):
        test_prediction(i, testing_data, y_test, w1, b1, w2, b2)


def init_weights():
    weights1 = randn(10, 784)
    bias1 = randn(10, 1)
    weights2 = randn(10, 10)
    bias2 = randn(10, 1)
    return weights1, bias1, weights2, bias2

def forward_propagtion(w1: ndarray, b1: ndarray, w2: ndarray, b2: ndarray, x):
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

def get_predictions(A2):
    return argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return sum(predictions == Y) / Y.size

def traning_loop(x: ndarray, y: ndarray, alpha: float, iterations: int):
    w1, b1, w2, b2 = init_weights()
    for i in range(iterations):
        one_hot_y = one_hot(y)
        z1, a1, z2, a2 = forward_propagtion(w1,b1,w2,b2,x)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, a2, w2, x, one_hot_y, x.shape[1])
        w1,w2,b1,b2 = update_parameters(w1, w2, b1,b2, dw1, db1, dw2, db2, alpha)
        if i % 10 == 0:
            print(f"Iteration: {i}")
            predictions = get_predictions(a2)
            print(get_accuracy(predictions, y))

    return w1, w2, b1, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagtion(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, x, y, W1, b1, W2, b2):
    current_image = x[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    main()
