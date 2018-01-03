import numpy as np
import h5py
from pylab import *

import matplotlib.pyplot as plt

from activation_functions import ReLU, sigmoid
from cost_functions import compute_cost_logistic
from neural_network import model_nn
from helper_functions import normalize_features


def get_data_sets():

    # Training Set
    train_path = 'datasets/cats/train_catvnoncat.h5'
    train_dataset = h5py.File(train_path, "r")
    X_train = np.array(train_dataset["train_set_x"][:])
    Y_train = np.array(train_dataset["train_set_y"][:])
    mtrain = X_train.shape[0]
    X_train = X_train.reshape(mtrain, -1)
    Y_train = Y_train.reshape(mtrain, 1)
    nx = X_train.shape[1]
    ny = Y_train.shape[1]

    # Test Set
    test_path = 'datasets/cats/test_catvnoncat.h5'
    test_dataset = h5py.File(test_path, "r")
    X_test = np.array(test_dataset["test_set_x"][:])
    Y_test = np.array(test_dataset["test_set_y"][:])
    mtest = X_test.shape[0]
    X_test = X_test.reshape(mtest, -1)
    Y_test = Y_test.reshape(mtest, 1)

    return X_train, Y_train, mtrain, \
           X_test,  Y_test,  mtest, \
           nx, ny


def main():
    # NN Settings
    learning_rate = 0.008
    # lambd = 0.00893
    lambd = 0.0

    # NN Architecture
    X_tr, Y_tr, m_tr, X_cv, Y_cv, m_cv, nx, ny = get_data_sets()
    activation_functions = {"g1": ReLU,
                            "g2": ReLU,
                            "g3": ReLU,
                            "g4": sigmoid}
    layer_dims = [nx, 20, 7, 5, ny]
    assert (len(activation_functions) == (len(layer_dims) - 1))

    # Iteration Settings
    min_iterations = 0
    max_iterations = 1300
    step_size = 20

    # Pre-process Data
    X_tr, X_cv = normalize_features(X_tr, X_cv)

    # Run
    parameters, data = model_nn(X_tr, Y_tr, X_cv, Y_cv,
                                learning_rate, lambd,
                                layer_dims, activation_functions, compute_cost_logistic,
                                min_iterations, max_iterations, step_size)

    # Unpack Data
    i_vals = data["i_vals"]
    cv_cost_vals = data["cv_cost_vals"]
    tr_cost_vals = data["tr_cost_vals"]
    accuracy_vals = data["accuracy_vals"]

    # Plot Results
    plt.plot(i_vals, cv_cost_vals, 'r')
    plt.plot(i_vals, tr_cost_vals, 'b')
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.title("lambda = " + str(lambd))
    plt.show()

    plt.plot(i_vals, accuracy_vals, 'g')
    plt.title("lambda = " + str(lambd))
    plt.show()


main()
