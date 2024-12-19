import numpy as np
from .train_val_split import *
from .plot import *
from src.activations import *
from src.layers import *
from src.learning_rate import *
from src.nn import *
from src.early_stopping import *
import time

# def grid_search_hold_out(train_data_in: np.ndarray, train_data_out: np.ndarray, eval_data_in: np.ndarray, eval_data_out: np.ndarray):

#     best_eval_loss = float('inf')
#     best_eval_losses = np.array([])
#     best_eval_accuracies = np.array([])
#     best_nn = None
#     input_size = train_data_in.shape[1]  # Number of input features
#     output_size = 1  # Number of output features

#     num_units = [2, 4, 6, 8]
#     num_layers = [1, 2]
#     act_fun = [Logistic, Tanh, ReLU, Leaky_ReLU]
#     learning_rates = [Learning_rate(0.01), Learning_rate(0.02), Linear_decay_learning_rate(0.02, 0.01, 100)]
#     regularization = [None, "Tikhonov", "Lasso"]
#     lambda_values = [0.00001, 0.0001, 0.001]
#     momentum_values = [None, 0.5, 0.9]
#     early_stopping = [None, Early_stopping(100, 0.0001), Early_stopping(100, 0.001)]
#     num_epochs = [700]

#     start_time = time.time()

#     for epochs in num_epochs:
#         for units in num_units:
#             for layers in num_layers:
#                 for activation in act_fun:
#                     for learning_rate in learning_rates:
#                         for reg in regularization:
#                             for lambda_par in lambda_values:
#                                 for momentum_par in momentum_values:
#                                     for stopping in early_stopping:
#                                         eval_loss_values = np.array([]) #Array che contiene ultima validation loss di ciascun trial
#                                         eval_losses = np.array([]) #Array di loss per un trial specifico
#                                         eval_accuracies = np.array([]) #Array di accuracy per un trial specifico
#                                         for i in range(0, 5):
#                                             layer_list = []
#                                             for i in range(layers + 1):
#                                                 in_size = input_size if i == 0 else units  # Input size for first layer
#                                                 out_size = units if i < layers - 1 else output_size  # Output size for last layer
#                                                 layer_list.append(Dense_layer(in_size, out_size, activation, "He"))
#                                             nn = FF_Neural_Network(input_size, layer_list, learning_rate, reg, lambda_par, momentum_par, stopping)
#                                             eval_losses, eval_accuracies = nn.train(train_data_in, train_data_out, epochs, eval_data_in, eval_data_out, 'Batch')
#                                             eval_loss_values = np.append(eval_loss_values, eval_losses[-1])
#                                         mean_loss = np.sum(eval_loss_values) / len(eval_loss_values)
#                                         print(mean_loss)
#                                         if (mean_loss < best_eval_loss):
#                                             best_eval_loss = mean_loss
#                                             best_eval_losses = eval_losses
#                                             best_eval_accuracies = eval_accuracies
#                                             best_nn = nn
    
#     print(f"Best eval loss: {best_eval_loss}")
#     print(f"Eval accuracy: {best_eval_accuracies[-1]}")
#     print(f"Obtained using model: {best_nn}")

#     end_time = time.time()
#     print(f"Elapsed time: {end_time - start_time}")

#     provaplot(best_eval_losses, best_eval_accuracies, len(best_eval_losses))

#     return best_nn

from itertools import product

def grid_search_hold_out(train_data_in: np.ndarray, train_data_out: np.ndarray, eval_data_in: np.ndarray, eval_data_out: np.ndarray):
    best_eval_loss = float('inf')
    best_eval_losses = np.array([])
    best_eval_accuracies = np.array([])
    best_nn = None
    input_size = train_data_in.shape[1]  # Number of input features
    output_size = 1  # Number of output features

    num_units = [2, 4, 6, 8]  # Possible number of units for hidden layers
    num_layers = [1, 2]  # Total number of layers (hidden + output)
    hidden_act_funs = [Logistic, Tanh, ReLU, Leaky_ReLU]  # Hidden layer activation functions
    output_act_funs = [Logistic, Tanh]  # Output layer activation functions
    learning_rates = [Learning_rate(0.01), Learning_rate(0.02), Linear_decay_learning_rate(0.02, 0.01, 100)]
    regularization = [None, "Tikhonov", "Lasso"]
    lambda_values = [0.00001, 0.0001, 0.001]
    momentum_values = [None, 0.5, 0.9]
    early_stopping = [None, Early_stopping(100, 0.0001), Early_stopping(100, 0.001)]
    num_epochs = [700]

    start_time = time.time()
    for epochs in num_epochs:
        for layers in num_layers:
        # Generate all combinations of units and activation functions for hidden layers
            hidden_layer_configs = list(product(num_units, hidden_act_funs))

            if layers == 1:
                # Only output layer (no hidden layers)
                for output_activation in output_act_funs:
                    for learning_rate in learning_rates:
                        for reg in regularization:
                            for lambda_par in lambda_values:
                                for momentum_par in momentum_values:
                                    for stopping in early_stopping:
                                        eval_loss_values = np.array([])
                                        eval_losses = np.array([])
                                        eval_accuracies = np.array([])

                                        for trial in range(5):  # 5 trials for each configuration
                                            # Create a network with only the output layer
                                            layer_list = [Dense_layer(input_size, output_size, output_activation, "He")]
                                            nn = FF_Neural_Network(input_size, layer_list, learning_rate, reg, lambda_par, momentum_par, stopping)
                                            eval_losses, eval_accuracies = nn.train(train_data_in, train_data_out, epochs, eval_data_in, eval_data_out, 'Batch')
                                            eval_loss_values = np.append(eval_loss_values, eval_losses[-1])

                                        mean_loss = np.sum(eval_loss_values) / len(eval_loss_values)
                                        if mean_loss < best_eval_loss:
                                            best_eval_loss = mean_loss
                                            best_eval_losses = eval_losses
                                            best_eval_accuracies = eval_accuracies
                                            best_nn = nn
            else:
                # Hidden layers + output layer
                for hidden_config in product(hidden_layer_configs, repeat=layers - 1):  # Hidden layers only
                    for output_activation in output_act_funs:
                        for learning_rate in learning_rates:
                            for reg in regularization:
                                for lambda_par in lambda_values:
                                    for momentum_par in momentum_values:
                                        for stopping in early_stopping:
                                            eval_loss_values = np.array([])
                                            eval_losses = np.array([])
                                            eval_accuracies = np.array([])

                                            for trial in range(5):  # 5 trials for each configuration
                                                layer_list = []
                                                # Build hidden layers dynamically
                                                for i, (units, activation) in enumerate(hidden_config):
                                                    in_size = input_size if i == 0 else hidden_config[i - 1][0]
                                                    layer_list.append(Dense_layer(in_size, units, activation, "He"))
                                                # Add output layer
                                                layer_list.append(Dense_layer(hidden_config[-1][0], output_size, output_activation, "He"))

                                                nn = FF_Neural_Network(input_size, layer_list, learning_rate, reg, lambda_par, momentum_par, stopping)
                                                eval_losses, eval_accuracies = nn.train(train_data_in, train_data_out, epochs, eval_data_in, eval_data_out, 'Batch')
                                                eval_loss_values = np.append(eval_loss_values, eval_losses[-1])

                                            mean_loss = np.sum(eval_loss_values) / len(eval_loss_values)
                                            if mean_loss < best_eval_loss:
                                                best_eval_loss = mean_loss
                                                best_eval_losses = eval_losses
                                                best_eval_accuracies = eval_accuracies
                                                best_nn = nn


    print(f"Best eval loss: {best_eval_loss}")
    print(f"Eval accuracy: {best_eval_accuracies[-1]}")
    print(f"Obtained using model: {best_nn}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    provaplot(best_eval_losses, best_eval_accuracies, len(best_eval_losses))

    return best_nn
