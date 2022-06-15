import numpy as np
from src.objects.Layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i], layers[i + 1]))

    def train(self, data, prediction):
        prediction_shape = prediction.shape

        if data.shape[0] != prediction.shape[0]:
            raise Exception('Data and prediction must have the same number of rows')

        for i in range(len(data)):
            input = np.array([data[i]])
            target = np.array([prediction[i]]).reshape(prediction_shape[1], -1)

            for i in range(len(self.layers)):
                input = self.layers[i].forward(input)

            for i in reversed(range(len(self.layers))):
                target = self.layers[i].backward(target, 0.1, i == len(self.layers) - 1)

    def predict(self, input):
        for i in range(len(self.layers)):
            input = self.layers[i].forward(input)
        return input

    def show(self):
        for i in range(len(self.layers)):
            print('weight', self.layers[i].weights)
            print('bias', self.layers[i].bias)
            print('inputs', self.layers[i].input)
            print('outputs', self.layers[i].output)
