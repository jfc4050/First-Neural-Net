import numpy as np
import math
np.seterr(all='raise')


def generate_minibatches(X_tr, Y_tr, minibatch_size):
    order = np.random.permutation(X_tr.shape[0])
    X_tr = np.take(X_tr, order, axis=0)
    Y_tr = np.take(Y_tr, order, axis=0)

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


def gradient_descent(layer, parameters, gradients, alpha):
    parameters["W" + str(layer)] -= alpha * gradients["dW" + str(layer)]
    parameters["b" + str(layer)] -= alpha * gradients["db" + str(layer)]


def gradient_descent_w_momentum(layer, parameters, gradients, velocities, α, β=0.9):
    l = str(layer)

    dW = gradients["dW" + l]
    vW = velocities["vW" + l]
    db = gradients["db" + l]
    vb = velocities["vb" + l]

    velocities["vW" + l] = β * vW + (1 - β) * dW
    velocities["vb" + l] = β * vb + (1 - β) * db

    parameters["W" + l] -= α * velocities["vW" + l]
    parameters["b" + l] -= α * velocities["vb" + l]


def ADAM(layer, parameters, gradients, velocities, t, α, β1=0.9, β2=0.999, ε=1e-9):
    l = str(layer)

    dW = gradients["dW" + l]
    db = gradients["db" + l]

    vW = velocities["vW" + l] ; vb = velocities["vb" + l]
    vW = (β1 * vW + (1 - β1) * dW) / (1 - (β1 ** t))
    vb = (β1 * vb + (1 - β1) * db) / (1 - (β1 ** t))
    velocities["vW" + l] = vW ; velocities["vb" + l] = vb

    sW = velocities["sW" + l] ; sb = velocities["sb" + l]
    sW = (β2 * sW + (1 - β2) * np.square(dW)) / (1 - (β2 ** t))
    sb = (β2 * sb + (1 - β2) * np.square(db)) / (1 - (β2 ** t))
    velocities["sW" + l] = sW ; velocities["sb" + l] = sb

    parameters["W" + l] -= α * np.divide(vW, np.sqrt(sW) + ε)
    parameters["b" + l] -= α * np.divide(vb, np.sqrt(sb) + ε)

