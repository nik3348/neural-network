import numpy as np


class Functions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_prime(x):
        # Alterantively: return sigmoid(x) * (1 - sigmoid(x))
        return np.exp(-x) / ((1 + np.exp(-x))**2)

    @staticmethod
    def back_propagation_error(output, target):
        if not isinstance(output, np.ndarray) or not isinstance(target, np.ndarray):
            raise Exception('Input must be a numpy array')
        elif target.shape[0] != output.shape[0]:
            raise Exception('Wrong input size')

        return 2 * (Functions.sigmoid(output) - target) * Functions.sigmoid_prime(output)

    # def error_function(y, y_hat):
    #     if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
    #         raise Exception('Input must be a numpy array')
    #     elif y.shape[0] != y_hat.shape[0] or y.shape[1] != y_hat.shape[1]:
    #         raise Exception('Wrong input size')

    #     return (y_hat - y)**2/2
