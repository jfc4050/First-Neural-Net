import numpy as np


def sigmoid(Z, derivative=False):
    if derivative:
        return np.multiply(sigmoid(Z), (1 - sigmoid(Z)))
    else:
        return 1 / (1 + np.exp(-Z))


def tanh(Z, derivative=False):
    if derivative:
        return 1 - np.power(np.tanh(Z), 2)
    else:
        return np.tanh(Z)


def ReLU(Z, derivative=False):
    if derivative:
        return Z > 0
    else:
        return np.multiply(Z, (Z > 0))
