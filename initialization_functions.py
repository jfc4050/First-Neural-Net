import numpy as np

def initialize_parameters(layer_dims, factor=2.):
    parameters = {}
    for i in range(1, len(layer_dims)):
        this_dim, last_dim = layer_dims[i], layer_dims[i-1]
        parameters["W" + str(i)] = np.random.randn(this_dim, last_dim) * np.sqrt(factor / last_dim)
        parameters["b" + str(i)] = np.zeros((this_dim, 1))

    return parameters


def initialize_velocities(layer_dims):
    velocities = {}
    for i in range(1, len(layer_dims)):
        this_dim, last_dim = layer_dims[i], layer_dims[i-1]
        velocities["vW" + str(i)] = np.zeros((this_dim, last_dim))
        velocities["sW" + str(i)] = np.zeros((this_dim, last_dim))
        velocities["vb" + str(i)] = np.zeros((this_dim, 1))
        velocities["sb" + str(i)] = np.zeros((this_dim, 1))

    return velocities