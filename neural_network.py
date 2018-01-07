from helper_functions import *
from initialization_functions import initialize_parameters, initialize_velocities
from propagation_functions import forwardpropagate, backpropagate
from cost_functions import compute_cost_logistic
from optimization_functions import *
from evaluation_functions import *


def model_nn(X_tr, Y_tr, X_cv, Y_cv,
             α, λ, minibatch_size,
             layer_dims, activation_functions, cost_function,
             epochs, step_size):

    # initialize empty data lists
    i_vals, \
    cv_cost_vals,      tr_cost_vals, \
    cv_accuracy_vals,  tr_accuracy_vals, \
    cv_precision_vals, tr_precision_vals, \
    cv_recall_vals,    tr_recall_vals, \
    cv_F1_vals,        tr_F1_vals = ([] for i in range(11))
    t = 0

    parameters = initialize_parameters(layer_dims)
    velocities = initialize_velocities(layer_dims)
    num_layers = len(activation_functions)

    mini_batches = generate_minibatches(X_tr, Y_tr, minibatch_size)

    for i in range(epochs):
        for mini_batch in mini_batches:
            X_tr, Y_tr = mini_batch
            outputs = forwardpropagate(X_tr, parameters, activation_functions, Y_tr, cost_function, λ)
            gradients = backpropagate(outputs, parameters, activation_functions, Y_tr, cost_function, λ)

            for layer in range(1, num_layers+1):
                t += 1
                # gradient_descent(layer, parameters, gradients, α)
                gradient_descent_w_momentum(layer, parameters, gradients, velocities, α)
                # ADAM(layer, parameters, gradients, velocities, t, α)

        if i % step_size == 0:
            # calculate and store data
            Yhat_tr = outputs["Ŷ"]
            Yhat_cv = get_Yhat(X_cv, parameters, activation_functions)
            cost_tr = outputs["J_unreg"]
            cost_cv = compute_cost_logistic(Yhat_cv, Y_cv)
            accuracy_tr = get_accuracy_logistic(Yhat_tr, Y_tr)
            accuracy_cv = get_accuracy_logistic(Yhat_cv, Y_cv)
            F1_tr, precision_tr, recall_tr = get_F1_logistic(Yhat_tr, Y_tr)
            F1_cv, precision_cv, recall_cv = get_F1_logistic(Yhat_cv, Y_cv)

            i_vals.append(i)
            tr_cost_vals.append(cost_tr) ; cv_cost_vals.append(cost_cv)
            tr_accuracy_vals.append(accuracy_tr) ; cv_accuracy_vals.append(accuracy_cv)
            tr_precision_vals.append(precision_tr) ; cv_precision_vals.append(precision_cv)
            tr_recall_vals.append(recall_tr) ; cv_recall_vals.append(recall_cv)
            tr_F1_vals.append(F1_tr) ; cv_F1_vals.append(F1_cv)

    data = {"i_vals": i_vals,
            "cv_cost_vals": cv_cost_vals, "tr_cost_vals": tr_cost_vals,
            "cv_accuracy_vals": cv_accuracy_vals, "tr_accuracy_vals": tr_accuracy_vals,
            "cv_precision_vals": cv_precision_vals, "tr_precision_vals": tr_precision_vals,
            "cv_recall_vals": cv_recall_vals, "tr_recall_vals": tr_recall_vals,
            "cv_F1_vals": cv_F1_vals, "tr_F1_vals": cv_F1_vals
            }

    return parameters, data