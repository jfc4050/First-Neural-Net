import numpy as np

def compute_cost_logistic(Yhat, Y, return_derivative=False):
    assert (Yhat.shape == Y.shape)

    if return_derivative:
        return np.divide(1 - Y, 1 - Yhat) - np.divide(Y, Yhat)
    else:
        m = Y.shape[0]
        return (-1 / m) * np.squeeze(np.sum(np.dot(Y.T, np.log(Yhat)) + np.dot((1-Y).T, np.log(1-Yhat))))