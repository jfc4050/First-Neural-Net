import numpy as np
import h5py

import matplotlib.pyplot as plt

from activation_functions import ReLU, sigmoid
from cost_functions import compute_cost_logistic
from neural_network import model_nn
from helper_functions import normalize_features


# def get_data_sets():
#
#     # Training Set
#     train_path = 'datasets/cats/train_catvnoncat.h5'
#     train_dataset = h5py.File(train_path, "r")
#     X_train = np.array(train_dataset["train_set_x"][:])
#     Y_train = np.array(train_dataset["train_set_y"][:])
#     mtrain = X_train.shape[0]
#     X_train = X_train.reshape(mtrain, -1)
#     Y_train = Y_train.reshape(mtrain, 1)
#     nx = X_train.shape[1]
#     ny = Y_train.shape[1]
#
#     # Test Set
#     test_path = 'datasets/cats/test_catvnoncat.h5'
#     test_dataset = h5py.File(test_path, "r")
#     X_test = np.array(test_dataset["test_set_x"][:])
#     Y_test = np.array(test_dataset["test_set_y"][:])
#     mtest = X_test.shape[0]
#     X_test = X_test.reshape(mtest, -1)
#     Y_test = Y_test.reshape(mtest, 1)
#
#     return X_train, Y_train, mtrain, \
#            X_test,  Y_test,  mtest, \
#            nx, ny

def get_data_sets():
    perc_train = 0.9
    path = 'datasets/cats/train_catvnoncat.h5'
    dataset = h5py.File(path, "r")
    X = np.array(dataset["train_set_x"][:])
    Y = np.array(dataset["train_set_y"][:])
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    assert(X.shape[0] == Y.shape[0])
    nX = X.shape[1]
    nY = Y.shape[1]

    # randomly shuffle dataset
    order = np.random.permutation(X.shape[0])
    m_tr = int(X.shape[0] * perc_train)
    tr_order = order[0:m_tr]
    cv_order = order[m_tr:]

    X_tr = np.take(X, tr_order, axis=0)
    Y_tr = np.take(Y, tr_order, axis=0)
    m_tr = X_tr.shape[0]

    X_cv = np.take(X, cv_order, axis=0)
    Y_cv = np.take(Y, cv_order, axis=0)
    m_cv = X_cv.shape[0]
    return X_tr, Y_tr, m_tr, \
           X_cv, Y_cv, m_cv, \
           nX, nY


def main():
    # NN Settings
    learning_rate = 0.003
    λ = 0.02

    # NN Architecture
    X_tr, Y_tr, m_tr, X_cv, Y_cv, m_cv, nX, nY = get_data_sets()
    activation_functions = {"g1": ReLU,
                            "g2": ReLU,
                            "g3": ReLU,
                            "g4": sigmoid}
    layer_dims = [nX, 20, 10, 5, nY]
    assert (len(activation_functions) == (len(layer_dims) - 1))

    # Iteration Settings
    minibatch_size = 64
    epochs = 600
    step_size = 5

    # Plotting Settings
    plotting_costs = True
    plotting_accuracy = True
    plotting_precision = False
    plotting_recall = False
    plotting_F1 = False

    # Pre-process Data
    X_tr, X_cv = normalize_features(X_tr, X_cv)

    # Run
    parameters, data = model_nn(X_tr, Y_tr, X_cv, Y_cv,
                                learning_rate, λ, minibatch_size,
                                layer_dims, activation_functions, compute_cost_logistic,
                                epochs, step_size)

    # Unpack Data
    i_vals = data["i_vals"]
    cv_cost_vals = data["cv_cost_vals"]
    tr_cost_vals = data["tr_cost_vals"]
    cv_accuracy_vals = data["cv_accuracy_vals"]
    tr_accuracy_vals = data["tr_accuracy_vals"]
    cv_precision_vals = data["cv_precision_vals"]
    tr_precision_vals = data["tr_precision_vals"]
    cv_recall_vals = data["cv_recall_vals"]
    tr_recall_vals = data["tr_recall_vals"]
    cv_F1_vals = data["cv_F1_vals"]
    tr_F1_vals = data["tr_F1_vals"]

    # Plot Results
    if plotting_costs:
        plt.plot(i_vals, cv_cost_vals, 'r')
        plt.plot(i_vals, tr_cost_vals, 'b')
        plt.xlabel("iterations")
        plt.ylabel("cost")
        plt.title("costs: lambda = " + str(λ))
        plt.show()

    if plotting_accuracy:
        plt.plot(i_vals, cv_accuracy_vals, 'r')
        plt.plot(i_vals, tr_accuracy_vals, 'b')
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.title("accuracy: lambda = " + str(λ))
        plt.show()

    if plotting_precision:
        plt.plot(i_vals, cv_precision_vals, 'r')
        plt.plot(i_vals, tr_precision_vals, 'b')
        plt.xlabel("iterations")
        plt.ylabel("precision")
        plt.title("precision: lambda = " + str(λ))
        plt.show()

    if plotting_recall:
        plt.plot(i_vals, cv_recall_vals, 'r')
        plt.plot(i_vals, tr_recall_vals, 'b')
        plt.xlabel("iterations")
        plt.ylabel("recall")
        plt.title("recall: lambda = " + str(λ))
        plt.show()

    if plotting_F1:
        plt.plot(i_vals, cv_F1_vals, 'r')
        plt.plot(i_vals, tr_F1_vals, 'b')
        plt.xlabel("iterations")
        plt.ylabel("F1")
        plt.title("F1: lambda = " + str(λ))
        plt.show()


main()
