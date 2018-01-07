import numpy as np
from propagation_functions import forwardpropagate


def normalize_features(x_train, x_cv):
    m = float(x_train.shape[0])
    mu = (1. / m) * np.sum(x_train, axis=0)
    sigma2 = (1. / m) * np.sum(np.square(x_train), axis=0)

    x_train = np.divide(x_train - mu, sigma2)
    x_cv = np.divide(x_cv - mu, sigma2)

    return x_train, x_cv


def get_Yhat(X, parameters, activation_functions):
    final_layer = len(activation_functions)
    output = forwardpropagate(X, parameters, activation_functions)

    return output["A" + str(final_layer)].T

