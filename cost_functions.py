import numpy as np

old_err_state = np.seterr(divide='raise')


def compute_cost_logistic(Ŷ, Y, derivative=False, λ=0., sum_weights=0.):
    assert(Ŷ.shape == Y.shape)

    if derivative:
        return np.divide(1.-Y, 1.-Ŷ) - np.divide(Y, Ŷ)
    elif not derivative:
        m = float(Y.shape[0])
        unregularized = (-1. / m) * np.squeeze(np.sum(np.dot(Y.T, np.log(Ŷ)) + np.dot((1.-Y).T, np.log(1.-Ŷ))))
        if λ == 0.0:
            return unregularized
        elif λ > 0.0:
            return unregularized + ((λ / (2*m)) * sum_weights)



