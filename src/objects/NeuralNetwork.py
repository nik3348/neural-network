import numpy as np
from objects.Layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list, learning_rate=1e-3, binary_regression=False):
        self.learning_rate = learning_rate
        self.binary_regression = binary_regression
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i], layers[i + 1]))

    def train(self, data, prediction):
        if data.shape[0] != prediction.shape[0]:
            raise Exception('Data and prediction must have the same number of rows')

        for i in range(len(data)):
            input = np.array([data[i]])
            target = np.array([prediction[i]])

            for x in range(len(self.layers)):
                has_softmax = (not self.binary_regression) and (x == (len(self.layers) - 1))
                input = self.layers[x].forward(input, has_softmax)

            for x in reversed(range(len(self.layers))):
                target = self.layers[x].backward(target, self.learning_rate, (x == (len(self.layers) - 1)))

    def predict(self, input):
        for i in range(len(self.layers)):
            has_softmax = (not self.binary_regression) and (i == (len(self.layers) - 1))
            input = self.layers[i].forward(input, has_softmax)
        return input

    def show(self):
        for i in range(len(self.layers)):
            print('weight', self.layers[i].weights)
            print('bias', self.layers[i].bias)
            print('inputs', self.layers[i].input)
            print('outputs', self.layers[i].output)
