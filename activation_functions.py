import numpy as np


def ReLU(Z, derivative=False):
    if derivative:
        return Z > 0
    else:
        return np.multiply(Z, (Z > 0))


def sigmoid(Z, derivative=False):
    if derivative:
        return np.multiply(sigmoid(Z), (1 - sigmoid(Z)))
    else:
        return np.power(1 + np.exp(-Z), -1)


def tanh(Z, derivative=False):
    if derivative:
        return 1 - np.power(np.tanh(Z), 2)
    else:
        return np.tanh(Z)


