import numpy as np
from propagation_functions import forwardpropagate


def normalize_features(X_train, X_cv):
    m = float(X_train.shape[0])
    mu = (1 / m) * np.sum(X_train, axis=0)
    sigma2 = (1 / m) * np.sum(np.square(X_train), axis=0)

    X_train = np.divide(X_train - mu, sigma2)
    X_cv = np.divide(X_cv - mu, sigma2)

    return X_train, X_cv


def initialize_parameters(layer_dims):
    parameters = {}
    for i in range(1, len(layer_dims)):
        this_dim = layer_dims[i]
        last_dim = layer_dims[i-1]
        parameters["W" + str(i)] = np.random.randn(this_dim, last_dim) * np.sqrt(2.0 / last_dim)
        parameters["b" + str(i)] = np.zeros((this_dim, 1))

    return parameters


def get_Yhat(X, parameters, activation_functions):
    final_layer = len(activation_functions)
    output = forwardpropagate(X, parameters, activation_functions)

    return output["A" + str(final_layer)].T


def get_accuracy_logistic(Yhat, Y):
    predictions = Yhat > 0.5
    correct = (predictions == Y)
    return np.mean(correct)
