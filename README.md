## A from scratch artificial neural network
> The goal of this project is to create a neural network without using any ML libraries, like PyTorch, to classify handwritten digits. And it was an experimental project to learn on how in reality neural networks work.

Through this documentation we're going first off to discuss on how to setup of this project in your local environment and later on I'm going to go guide you how I implemented this project also we will learn what artificial neural networks are. 
As for the dataset to train the neural network I use the **MNIST** dataset.

## Quickstart
### Prerequisites
- **Python:** 3.14+
- One of:
    - **uv** (recommended)
    - **pip + venv** 

Clone the github repo(or download it):
```
git clone https://github.com/gazi04/digit-recognition.git
```

Go to the project directory and run:
```
uv sync
uv run main.py
```

## The MATH!
> DISCLAIMER: Is prefered to posses knowledge about what matrixes, matrix operations are cause this whole project is surrounded by linear algebra and calculus.

Now we are going to start with the fun part but first, "What is a neural network?" essentially it's just a bunch of chained mathematical functions that takes an input and gives an output that's all and the most interesting part is that is some sense they learn like a human that's the reason why they're called artificial neural network. The way on how they were designed was inspired by the human brain itself.

### The mental model of neural networks (data structure)
Maybe you already know from different sources that a neural network has layers there are the input layer, the hidden layer and the output layer. Think about those layers like they are matrices, if layer **A** has 3 neurons and there is another layer **B** that has 2 neurons the connection between them would be a matrix 3x2.

### Initialization
Those matrices are going to be filled with numbers (called **weights** and **biases**) that we're going to fine-tune until we get the desired output, this is the first step. You can't start with the weights at zero cause the way how the math fucntions are defined they aren't going to work, in other words it breaks the math.

This first step is called **weight initialization** or **parameter initialization**, there are a lot of methods on how to initiate those weights (or the network, cause those weights builds the neural network *just wanted to mention this*) like **He initialization**, **Glorot initialization** etc.

And this is how the initialization will look like, mine is very simple I just use this `randn()` function from numpy. You could use a more sphophisticated approach.

```python
def init_weights():
    weights1 = randn(10, 784)
    bias1 = randn(10, 1)
    weights2 = randn(10, 10)
    bias2 = randn(10, 1)
    return weights1, bias1, weights2, bias2
```

#### Code explaination
- The input layer (784 nodes): First off we have the input layer that's the `784` value on the second line. The images in the **MNIST** dataset contain 28x28 images so and if we do the math there are 784 pixel for an image.

- The hidden layer (10 neurons): The whole image now is going to be feeded inside a single neuron, in the decimal number system there are 10 numbers `0,1,2,3,4,5,6,7,8,9` so this is the reason why there is `10` in this line `weights1 = randn(10, 784)`. And lastly about the `bias1` this is just an array of numbers for each neuron.

- The output layer (10 neurons): 
Every pixel is going to be connected with all of the 10 neurons and lastly those 10 neurons are going to be conntected withe 


### Forward propagation
$$\mathbf{Z}^{[1]}=\mathbf{W}^{[1]}X+\mathbf{b}^{[1]}$$
$$\mathbf{A}^{[1]}=g_{ReLU}(Z^{[1]})$$
$$\mathbf{Z}^{[2]}=\mathbf{W}^{[2]}{A}^{[1]}+b^{[2]}$$
$$\mathbf{A}^{[2]}=\mathbf{g}_{softmax}(Z^{[2]})$$

### Backward propagation
$$dZ^{[2]}=A^{[2]}-Y$$
$$dW^{[2]}=\frac{1}{m}dZ^{[2]}A{[1]T}$$
$$dB^{[2]}=\frac{1}{m}\Sigma{dZ}^{[2]}$$
$$dZ^{[1]}=W^{[2]T}dZ^{[2]}*\dot{g^{[1]}}(z^{[1]})$$
$$dW^{[1]}=\frac{1}{m}dZ^{[1]}A^{[0]T}$$
$$dB^{[1]}=\frac{1}{m}\Sigma{dZ}^{[1]}$$

### Parameter updates

$$W^{[2]}:=W^{[2]}-\alpha{dW}^{[2]}$$
$$b^{[2]}:=b^{[2]}-\alpha{db}^{[2]}$$
$$W^{[1]}:=W^{[1]}-\alpha{dW}^{[1]}$$
$$b^{[1]}:=b^{[1]}-\alpha{db}^{[1]}$$
