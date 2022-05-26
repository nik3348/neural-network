from ai_wine_quality_regression.objects.Layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i], layers[i + 1]))

    def train(self, input, target):
        for i in range(len(self.layers)):
            input = self.layers[i].forward(input)

        for i in reversed(range(len(self.layers))):
            target = self.layers[i].backward(target, 1)

    def predict(self, input):
        for i in range(len(self.layers)):
            input = self.layers[i].forward(input)
        return input
