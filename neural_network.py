from propagation_functions import forwardpropagate, backpropagate
from helper_functions import get_Yhat, get_accuracy_logistic, initialize_parameters, generate_minibatches
from cost_functions import compute_cost_logistic


def model_nn(X_tr, Y_tr, X_cv, Y_cv,
             learning_rate, lambd, minibatch_size,
             layer_dims, activation_functions, cost_function,
             min_iterations, max_iterations, step_size):

    i_vals, cv_cost_vals, tr_cost_vals, accuracy_vals = ([] for i in range(4))

    parameters = initialize_parameters(layer_dims)
    num_layers = len(activation_functions)

    mini_batches = generate_minibatches(X_tr, Y_tr, minibatch_size)

    for i in range(min_iterations, max_iterations):
        for mini_batch in mini_batches:
            X_tr, Y_tr = mini_batch
            outputs = forwardpropagate(X_tr, parameters, activation_functions, Y_tr, cost_function, lambd)
            gradients = backpropagate(outputs, parameters, activation_functions, Y_tr, cost_function, lambd)

            for layer in range(1, num_layers+1):
                parameters["W" + str(layer)] -= learning_rate * gradients["dW" + str(layer)]
                parameters["b" + str(layer)] -= learning_rate * gradients["db" + str(layer)]

        if i % step_size == 0:
            Yhat_tr = outputs["Ŷ"]
            Yhat_cv = get_Yhat(X_cv, parameters, activation_functions)
            cost_tr = outputs["J_unreg"]
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