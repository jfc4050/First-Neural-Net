import numpy as np
import math
from propagation_functions import forwardpropagate


def normalize_features(x_train, x_cv):
    m = float(x_train.shape[0])
    mu = (1. / m) * np.sum(x_train, axis=0)
    sigma2 = (1. / m) * np.sum(np.square(x_train), axis=0)

    x_train = np.divide(x_train - mu, sigma2)
    x_cv = np.divide(x_cv - mu, sigma2)

    return x_train, x_cv


def initialize_parameters(layer_dims, factor=2.):
    parameters = {}
    for i in range(1, len(layer_dims)):
        this_dim = layer_dims[i]
        last_dim = layer_dims[i-1]
        parameters["W" + str(i)] = np.random.randn(this_dim, last_dim) * np.sqrt(factor / last_dim)
        parameters["b" + str(i)] = np.zeros((this_dim, 1))

    return parameters


def generate_minibatches(X_tr, Y_tr, minibatch_size):
    # TODO: Shuffle
    m = X_tr.shape[0]
    s = minibatch_size
    num_complete_minibatches = math.floor(m / s)
    minibatches = []
    for k in range(num_complete_minibatches):
        X_m = X_tr[k*s:(k+1)*s, :]
        Y_m = Y_tr[k*s:(k+1)*s, :]
        minibatches.append((X_m, Y_m))
    if m % minibatch_size != 0:
        X_m = X_tr[num_complete_minibatches*s:, :]
        Y_m = Y_tr[num_complete_minibatches*s:, :]
        minibatches.append((X_m, Y_m))

    return minibatches



def get_Yhat(X, parameters, activation_functions):
    final_layer = len(activation_functions)
    output = forwardpropagate(X, parameters, activation_functions)

    return output["A" + str(final_layer)].T


def get_accuracy_logistic(Yhat, Y):
    predictions = Yhat > 0.5
    correct = (predictions == Y)
    return np.mean(correct)
