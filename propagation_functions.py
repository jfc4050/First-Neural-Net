import numpy as np


def forwardpropagate(X, parameters, activation_functions,
                     Y=None, cost_function=None, λ=0.):
    last_layer = len(activation_functions)
    outputs = {"A0": X.T}
    sum_weights = 0.

    def forwardpropagate_recurse(outputs, sum_weights, layer=1):
        prev_A = outputs["A" + str(layer - 1)]
        g = activation_functions["g" + str(layer)]
        W = parameters["W" + str(layer)]
        b = parameters["b" + str(layer)]

        Z = np.dot(W, prev_A) + b
        A = g(Z)
        outputs["A" + str(layer)] = A
        outputs["Z" + str(layer)] = Z
        sum_weights += np.sum(W)

        if layer is not last_layer:
            return forwardpropagate_recurse(outputs, sum_weights, layer+1)
        elif layer is last_layer:
            Ŷ = A.T
            outputs["Ŷ"] = Ŷ
            if cost_function is not None:
                assert(Y is not None)
                outputs["J_unreg"] = cost_function(Ŷ, Y, λ=0, sum_weights=sum_weights)
                outputs["J_reg"]   = cost_function(Ŷ, Y, λ=λ, sum_weights=sum_weights)

        return outputs

    return forwardpropagate_recurse(outputs, sum_weights)


def backpropagate(outputs, parameters, activation_functions, Y, cost_function, lambd=0.0):
    m = float(Y.shape[0])
    last_layer = len(activation_functions)
    gradients = {}

    def backpropagate_recurse(layer=1):

        if layer is not last_layer:
            this_dA = backpropagate_recurse(layer + 1)
        elif layer is last_layer:
            Yhat = outputs["Ŷ"]
            this_dA = cost_function(Yhat, Y, derivative=True).T

        this_g = activation_functions["g" + str(layer)]
        this_W = parameters["W" + str(layer)]
        this_Z = outputs["Z" + str(layer)]
        prev_A = outputs["A" + str(layer - 1)]

        this_dZ = np.multiply(this_dA, this_g(this_Z, derivative=True))
        this_dW = ( (1.0 / m) * np.dot(this_dZ, prev_A.T) )
        this_db = (1.0 / m) * np.sum(this_dZ, axis=1, keepdims=True)
        gradients["dW" + str(layer)] = this_dW + ( (lambd / m) * this_W )
        gradients["db" + str(layer)] = this_db

        return np.dot(this_W.T, this_dZ)  # return prev_dA

    backpropagate_recurse()
    return gradients
