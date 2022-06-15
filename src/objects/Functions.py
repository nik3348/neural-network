import numpy as np


class Functions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return np.exp(-x) / ((1 + np.exp(-x))**2)

    @staticmethod
    def sigmoided_prime(x):
        return x * (1 - x)

    @staticmethod
    def back_propagation_error(sigmoided_output, target):
        return np.multiply((sigmoided_output - target), Functions.sigmoided_prime(sigmoided_output))
