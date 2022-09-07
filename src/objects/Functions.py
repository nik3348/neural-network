import numpy as np


class Functions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return np.exp(-x) / ((1 + np.exp(-x))**2)

    @staticmethod
    def sigmoided_prime(dx):
        return dx * (1 - dx)

    @staticmethod
    def back_propagation_error(sigmoided_output, target):
        return np.multiply((sigmoided_output - target), Functions.sigmoided_prime(sigmoided_output))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_prime(x):
        return 1 * (x>0)

    @staticmethod
    def leaky_relu(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_prime(x, alpha=0.1):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def softmax(x):
        return(x/np.sum(x))
