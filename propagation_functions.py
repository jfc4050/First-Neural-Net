import numpy as np


def forwardpropagate(X, parameters, activation_functions):
    last_layer = len(activation_functions)

    def forwardpropagate_recurse(outputs, layer=1):
        prev_A = outputs["A" + str(layer - 1)]
        g = activation_functions["g" + str(layer)]
        W = parameters["W" + str(layer)]
        b = parameters["b" + str(layer)]

        Z = np.dot(W, prev_A) + b
        outputs["A" + str(layer)] = g(Z)
        outputs["Z" + str(layer)] = Z

        if layer != last_layer:
            return forwardpropagate_recurse(outputs, layer + 1)

        return outputs

    outputs = {"A0": X.T}
    return forwardpropagate_recurse(outputs)


def backpropagate(outputs, parameters, activation_functions, Y, cost_function):
    m = Y.shape[0]
    last_layer = len(activation_functions)
    gradients = {}

    def backpropagate_recurse(layer=1):

        if layer != last_layer:
            this_dA = backpropagate_recurse(layer + 1)
        else:  # this is the last layer
            Yhat = outputs["A" + str(layer)].T
            this_dA = cost_function(Yhat, Y, return_derivative=True).T


        this_dg = activation_functions["g" + str(layer)]
        this_W = parameters["W" + str(layer)]
        this_Z = outputs["Z" + str(layer)]
        prev_A = outputs["A" + str(layer - 1)]

        this_dZ = np.multiply(this_dA, this_dg(this_Z, return_derivative=True))
        gradients["dW" + str(layer)] = (1.0 / m) * np.dot(this_dZ, prev_A.T)
        gradients["db" + str(layer)] = (1.0 / m) * np.sum(this_dZ, axis=1, keepdims=True)

        return np.dot(this_W.T, this_dZ)  # return prev_dA

    backpropagate_recurse()
    return gradients
