import numpy as np
from propagation_functions import forwardpropagate

def initialize_parameters(layer_dims, scale_term=0.01):
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * scale_term
        parameters["b" + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters


def get_Yhat(X, parameters, activation_functions):
    final_layer = len(activation_functions)
    output = forwardpropagate(X, parameters, activation_functions)

    return output["A" + str(final_layer)].T