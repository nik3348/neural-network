import copy
import numpy as np

from objects.Functions import Functions


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size) # The 1 makes this a 2d array instead of a 1d array
        self.weights = self.weights.astype('float128')
        self.bias = self.bias.astype('float128')

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        self.sigmoid = Functions.sigmoid(self.output)
        return self.sigmoid

    def backward(self, target, learning_rate, is_output=False):
        if is_output:
            error = self.sigmoid - target
        else:
            error = target

        output_error = np.multiply(error, Functions.sigmoided_prime(self.sigmoid))

        weights_gradient = np.dot(self.input.T, output_error)
        original_weights = copy.copy(self.weights)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_error

        return np.dot(output_error, original_weights.T)
