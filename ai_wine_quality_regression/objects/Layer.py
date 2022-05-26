import numpy as np

from objects.Functions import Functions


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size, 1) # The 1 makes this a 2d array instead of a 1d array

    def forward(self, input):
        if not isinstance(input, np.ndarray):
            raise Exception('Input must be a numpy array')
        elif input.shape[0] != self.weights.shape[1]:
            raise Exception('Wrong input size')

        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias
        return Functions.sigmoid(self.output)

    def backward(self, target, learning_rate):
        output_gradient = Functions.back_propagation_error(self.output, target)
        weights_gradient = np.dot(output_gradient, self.input.T)
        original_weights = self.weights

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return np.dot(original_weights.T, output_gradient)
