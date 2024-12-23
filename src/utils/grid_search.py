import numpy as np
from .data_split import *
from .plot import *
from src.activations import *
from src.layers import *
from src.learning_rate import *
from src.nn import *
from src.momentum import *
from src.early_stopping import *
import time
from itertools import product

def grid_search_hold_out(
        train_data_in: np.ndarray,
        train_data_out: np.ndarray,
        eval_data_in: np.ndarray,
        eval_data_out: np.ndarray,
        units: List[int],
        layers: List[int],
        activations: List[ActivationFunction],
        l_rates: List[Learning_rate],
        regularizers: List[str],
        lambdas: List[float],
        momentums: List[Momentum],
        stoppings: List[Early_stopping],
        epochs: List[int]
        ):
    
    best_eval_loss = float('inf')
    best_eval_standard_deviation = float('inf')
    best_eval_losses = np.array([])
    best_eval_accuracies = np.array([])
    best_nn = None
    input_size = train_data_in.shape[1]  # Number of input features
    output_size = train_data_out.shape[1] if train_data_out.ndim != 1 else 1  # Number of output features

    # Options for each layer
    num_units = units  # Possible number of units for hidden layers
    num_layers = layers
    act_funs = activations  # Hidden layer activation functions
    learning_rates = l_rates
    regularization = regularizers
    lambda_values = lambdas
    momentum_values = momentums
    early_stopping = stoppings
    num_epochs = epochs

    layer_configs = all_layer_configs(num_units, num_layers, act_funs, input_size, output_size)
    print(len(layer_configs))

    start_time = time.time()
    counter = 0
    for learning_rate in learning_rates:
        for reg in regularization:
            for lambda_par in lambda_values:
                if (reg is None and lambda_par is not None):
                    continue
                if (reg is not None and lambda_par is None):
                    continue
                for momentum in momentum_values:
                    for stopping in early_stopping:
                        for epochs in num_epochs:
                            for config in layer_configs:
                                eval_loss_values = np.array([])
                                eval_losses = np.array([])
                                eval_accuracies = np.array([])

                                nn = FF_Neural_Network(input_size, config, learning_rate, reg, lambda_par, momentum, stopping)
                                for trial in range(5):
                                    eval_losses, eval_accuracies = nn.train(train_data_in, train_data_out, epochs, False, eval_data_in, eval_data_out)
                                    eval_loss_values = np.append(eval_loss_values, eval_losses[-1])
                                    #print(eval_losses[-1])
                                    nn.reset()
                                mean_loss = np.mean(eval_loss_values)
                                print(f"MEAN LOSS: {mean_loss}")
                                mean_standard_deviation = np.std(eval_loss_values)
                                if mean_loss < best_eval_loss:
                                    print(f"UPDATED BEST LOSS: {mean_loss}")
                                    best_eval_loss = mean_loss
                                    best_eval_standard_deviation = mean_standard_deviation
                                    best_eval_losses = eval_losses
                                    best_eval_accuracies = eval_accuracies
                                    best_nn = nn
                                counter += 1
                                if (counter % 100 == 0):
                                    print(f"CONFIGURATION #{counter} TRAINED")


    print(f"Best eval loss: {best_eval_loss}")
    print(f"Eval standard deviation: {best_eval_standard_deviation}")
    print(f"Eval accuracy: {best_eval_accuracies[-1]}")
    print(f"Obtained using model: {best_nn}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    provaplot(best_eval_losses, best_eval_accuracies, len(best_eval_losses))

    return best_nn, best_eval_loss

def grid_search_k_fold(
        input_data: np.ndarray,
        output_data: np.ndarray,
        units: List[int],
        layers: List[int],
        activations: List[ActivationFunction],
        l_rates: List[Learning_rate],
        regularizers: List[str],
        lambdas: List[float],
        momentums: List[Momentum],
        stoppings: List[Early_stopping],
        epochs: List[int]
        ):
    best_eval_loss = float('inf')
    best_eval_standard_deviation = float('inf')
    best_eval_losses = np.array([])
    best_eval_accuracies = np.array([])
    best_nn = None
    input_size = input_data[0].shape[1]  # Number of input features
    output_size = output_data[0].shape[1] if output_data[0].ndim != 1 else 1  # Number of output features

    # Options for each layer
    num_units = units  # Possible number of units for hidden layers
    num_layers = layers
    act_funs = activations  # Hidden layer activation functions
    learning_rates = l_rates
    regularization = regularizers
    lambda_values = lambdas
    momentum_values = momentums
    early_stopping = stoppings
    num_epochs = epochs

    layer_configs = all_layer_configs(num_units, num_layers, act_funs, input_size, output_size)
    print(len(layer_configs))

    start_time = time.time()
    counter = 0
    for learning_rate in learning_rates:
        for reg in regularization:
            for lambda_par in lambda_values:
                if (reg is None and lambda_par is not None):
                    continue
                if (reg is not None and lambda_par is None):
                    continue
                for momentum in momentum_values:
                    for stopping in early_stopping:
                        for epochs in num_epochs:
                            for config in layer_configs:
                                eval_loss_values = np.array([]) #Eval losses over all folds

                                nn = FF_Neural_Network(input_size, config, learning_rate, reg, lambda_par, momentum, stopping)
                                for i, fold in enumerate(input_data):
                                    fold_loss_values = np.array([]) #losses on different trials over a fold
                                    eval_losses = np.array([]) #Losses over same trials over the same fold
                                    eval_accuracies = np.array([])
                                    train_data_in = np.concatenate([input_data[j] for j in range(len(input_data)) if j != i])
                                    train_data_out = np.concatenate([output_data[j] for j in range(len(input_data)) if j != i])
                                    eval_data_in = fold
                                    eval_data_out = output_data[i]
                                    for trial in range(3):
                                        eval_losses, eval_accuracies = nn.train(train_data_in, train_data_out, epochs, False, eval_data_in, eval_data_out, 'Batch')
                                        fold_loss_values = np.append(fold_loss_values, eval_losses[-1])
                                        nn.reset()
                                    mean_fold_loss = np.mean(fold_loss_values)
                                    eval_loss_values = np.append(eval_loss_values, mean_fold_loss)
                                
                                mean_loss = np.mean(eval_loss_values)
                                print(f"MEAN LOSS: {mean_loss}")
                                mean_standard_deviation = np.std(eval_loss_values)
                                if mean_loss < best_eval_loss:
                                    print(f"UPDATED BEST LOSS: {mean_loss}")
                                    best_eval_loss = mean_loss
                                    best_eval_standard_deviation = mean_standard_deviation
                                    best_eval_losses = eval_losses
                                    best_eval_accuracies = eval_accuracies
                                    best_nn = nn
                                counter += 1
                                if (counter % 100 == 0):
                                    print(f"CONFIGURATION #{counter} TRAINED")

    print(f"Best eval loss: {best_eval_loss}")
    print(f"Eval standard diviation: {best_eval_standard_deviation}")
    print(f"Eval accuracy: {best_eval_accuracies[-1]}")
    print(f"Obtained using model: {best_nn}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    provaplot(best_eval_losses, best_eval_accuracies, len(best_eval_losses))

    return best_nn, best_eval_loss

def all_layer_configs(num_units: List[int], num_layers: List[int], act_funs: list[ActivationFunction], input_size: int, output_size: int):
    unit_act_configs = list()
    for i in range(len(num_layers)):
        units = list(product(num_units, repeat = i + 1))
        activations = list((product(act_funs, repeat = i + 1)))
        prod = list(product(units, activations))
        unit_act_configs += prod

    layer_configs = []
    for config in unit_act_configs:
        layer_list = [Dense_layer(input_size, config[0][0], config[1][0])] #Input layer
        for i, num_units in enumerate(config[0]):
            if (i == len(config[0]) - 1): #output layer
                layer = Dense_layer(num_units, output_size, Logistic)
            else: #Hidden layer
                layer = Dense_layer (num_units, config[0][i + 1], config[1][i + 1])
            layer_list.append(layer)
        layer_configs.append(layer_list)
    
    return layer_configs