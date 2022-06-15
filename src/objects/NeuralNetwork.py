import numpy as np
from src.objects.Layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i], layers[i + 1]))

    def train(self, data, prediction):
        data_shape = data.shape
        prediction_shape = prediction.shape

        if data.shape[0] != prediction.shape[0]:
            raise Exception('Data and prediction must have the same number of rows')

        for i in range(len(data)):
            input = np.array([data[i]])
            target = np.array([prediction[i]]).reshape(prediction_shape[1], -1)

            for i in range(len(self.layers)):
                input = self.layers[i].forward(input)

            for i in reversed(range(len(self.layers))):
                target = self.layers[i].backward(target, 0.001)

    def predict(self, input):
        for i in range(len(self.layers)):
            input = self.layers[i].forward(input)
        return input

    def show(self):
        for i in range(len(self.layers)):
            print('nweight', self.layers[i].weights)
            print('nbias', self.layers[i].bias)
            print('ninputs', self.layers[i].input)
            print('noutputs', self.layers[i].output)
