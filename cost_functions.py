import numpy as np

old_err_state = np.seterr(divide='raise')


def compute_cost_logistic(Yhat, Y, derivative=False):
    assert (Yhat.shape == Y.shape)

    if derivative:
        return np.divide(1 - Y, 1 - Yhat) - np.divide(Y, Yhat)
    else:
        m = float(Y.shape[0])
        return (-1 / m) * np.squeeze(np.sum(np.dot(Y.T, np.log(Yhat)) + np.dot((1-Y).T, np.log(1-Yhat))))
