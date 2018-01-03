from propagation_functions import forwardpropagate, backpropagate
from helper_functions import get_Yhat, get_accuracy_logistic, initialize_parameters
from cost_functions import compute_cost_logistic


def model_nn(X_tr, Y_tr, X_cv, Y_cv,
             learning_rate, lambd,
             layer_dims, activation_functions, cost_function,
             min_iterations, max_iterations, step_size):

    i_vals, cv_cost_vals, tr_cost_vals, accuracy_vals = ([] for i in range(4))

    parameters = initialize_parameters(layer_dims)
    num_layers = len(activation_functions)

    for i in range(min_iterations, max_iterations):
        outputs = forwardpropagate(X_tr, parameters, activation_functions)
        gradients = backpropagate(outputs, parameters, activation_functions, Y_tr, cost_function, lambd)

        for layer in range(1, num_layers+1):
            parameters["W" + str(layer)] -= learning_rate * gradients["dW" + str(layer)]
            parameters["b" + str(layer)] -= learning_rate * gradients["db" + str(layer)]

        if i % step_size == 0:
            Yhat_tr = get_Yhat(X_tr, parameters, activation_functions)
            Yhat_cv = get_Yhat(X_cv, parameters, activation_functions)
            cost_tr = compute_cost_logistic(Yhat_tr, Y_tr)
            cost_cv = compute_cost_logistic(Yhat_cv, Y_cv)
            accuracy = get_accuracy_logistic(Yhat_cv, Y_cv)

            i_vals.append(i)
            cv_cost_vals.append(cost_cv)
            tr_cost_vals.append(cost_tr)
            accuracy_vals.append(accuracy)

    data = {"i_vals": i_vals,
            "cv_cost_vals": cv_cost_vals,
            "tr_cost_vals": tr_cost_vals,
            "accuracy_vals": accuracy_vals}

    return parameters, data