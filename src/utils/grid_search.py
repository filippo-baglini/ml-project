import numpy as np
import random
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
        num_units: List[int],
        num_layers: List[int],
        activations: List[ActivationFunction],
        learning_rates: List[Learning_rate],
        regularization: List[str],
        lambda_values: List[float],
        momentum_values: List[Momentum],
        early_stopping: List[Early_stopping],
        num_epochs: List[int],
        task: Optional[Literal["Classification", "Regression"]] = "Classification"
        ):
    
    best_eval_loss = float('inf')
    best_train_loss = float('inf')
    best_eval_standard_deviation = float('inf')
    best_eval_losses = np.array([])
    best_eval_accuracies = np.array([])
    best_nn = None
    input_size = train_data_in.shape[1]  # Number of input features
    output_size = train_data_out.shape[1] if train_data_out.ndim != 1 else 1  # Number of output features

    layer_configs = all_layer_configs(num_units, num_layers, activations, input_size, output_size, task)
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
                                train_loss_values = np.array([])
                                eval_losses = []
                                eval_accuracies = []

                                nn = FF_Neural_Network(input_size, config, learning_rate, reg, lambda_par, momentum, stopping)

                                for trial in range(5):
                                    eval_loss, eval_accuracy, train_loss = nn.train(train_data_in, train_data_out, epochs, False, eval_data_in, eval_data_out)
                                    eval_losses.append(eval_loss)
                                    eval_accuracies.append(eval_accuracy)
                                    eval_loss_values = np.append(eval_loss_values, eval_loss[-1])
                                    train_loss_values = np.append(train_loss_values, train_loss)
                                    nn.reset()
                                mean_eval_loss = np.mean(eval_loss_values)
                                mean_train_loss = np.mean(train_loss_values)
                                print(f"MEAN LOSS: {mean_eval_loss}")
                                mean_standard_deviation = np.std(eval_loss_values)
                                if mean_eval_loss < best_eval_loss:
                                    print(f"UPDATED BEST LOSS: {mean_eval_loss}")
                                    best_eval_loss = mean_eval_loss
                                    best_train_loss = mean_train_loss
                                    best_eval_standard_deviation = mean_standard_deviation
                                    best_eval_losses = eval_losses
                                    best_eval_accuracies = eval_accuracies
                                    best_nn = nn
                                counter += 1
                                if (counter % 100 == 0):
                                    print(f"CONFIGURATION #{counter} TRAINED")


    print(f"Best mean eval loss: {best_eval_loss}")
    print(f"Eval standard deviation: {best_eval_standard_deviation}")
    if (task == "Classification"):
        mean_accuracy = 0
        for run in best_eval_accuracies:
            mean_accuracy += run[-1]
        print(f"Mean eval accuracy: {mean_accuracy / len(best_eval_accuracies)}")
    print(f"Best mean train loss: {best_train_loss}")
    print(f"Obtained using model: {best_nn}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    if (task == "Classification"):
        plot_cross_validation(best_eval_losses, best_eval_accuracies)
    else:
        plot_loss_cross_validation(best_eval_losses)

    return best_nn, best_train_loss

def grid_search_k_fold(
        input_data: np.ndarray,
        output_data: np.ndarray,
        num_units: List[int],
        num_layers: List[int],
        activations: List[ActivationFunction],
        learning_rates: List[Learning_rate],
        regularization: List[str],
        lambda_values: List[float],
        momentum_values: List[Momentum],
        early_stopping: List[Early_stopping],
        num_epochs: List[int],
        task: Optional[Literal["Classification", "Regression"]] = "Classification"
        ):
    best_eval_loss = float('inf')
    best_train_loss = float('inf')
    best_eval_standard_deviation = float('inf')
    best_eval_losses = np.array([])
    best_eval_accuracies = np.array([])
    best_nn = None
    input_size = input_data[0].shape[1]  # Number of input features
    output_size = output_data[0].shape[1] if output_data[0].ndim != 1 else 1  # Number of output features

    layer_configs = all_layer_configs(num_units, num_layers, activations, input_size, output_size, task)
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
                                all_eval_losses = []
                                all_eval_accuracies = []
                                train_loss_values = np.array([])

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
                                        eval_losses, eval_accuracies, train_loss = nn.train(train_data_in, train_data_out, epochs, False, eval_data_in, eval_data_out)
                                        all_eval_losses.append(eval_losses)
                                        all_eval_accuracies.append(eval_accuracies)
                                        fold_loss_values = np.append(fold_loss_values, eval_losses[-1])
                                        train_loss_values = np.append(train_loss_values, train_loss)
                                        nn.reset()
                                    mean_fold_eval_loss = np.mean(fold_loss_values)
                                    eval_loss_values = np.append(eval_loss_values, mean_fold_eval_loss)
                                
                                mean_eval_loss = np.mean(eval_loss_values)
                                print(f"MEAN LOSS: {mean_eval_loss}")
                                mean_standard_deviation = np.std(eval_loss_values)
                                mean_train_loss = np.mean(train_loss_values)
                                if mean_eval_loss < best_eval_loss:
                                    print(f"UPDATED BEST LOSS: {mean_eval_loss}")
                                    best_eval_loss = mean_eval_loss
                                    best_eval_standard_deviation = mean_standard_deviation
                                    best_train_loss = mean_train_loss
                                    best_eval_losses = all_eval_losses
                                    best_eval_accuracies = all_eval_accuracies
                                    best_nn = nn
                                counter += 1
                                if (counter % 100 == 0):
                                    print(f"CONFIGURATION #{counter} TRAINED")

    print(f"Best mean eval loss: {best_eval_loss}")
    print(f"Eval standard deviation: {best_eval_standard_deviation}")
    if (task == "Classification"):
        mean_accuracy = 0
        for run in best_eval_accuracies:
            mean_accuracy += run[-1]
        print(f"Mean eval accuracy: {mean_accuracy / len(best_eval_accuracies)}")
    print(f"Best mean train loss: {best_train_loss}")
    print(f"Obtained using model: {best_nn}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    if (task == "Classification"):
        plot_cross_validation(best_eval_losses, best_eval_accuracies)
    else:
        plot_loss_cross_validation(best_eval_losses)

    return best_nn, best_train_loss


def random_search_hold_out(
        train_data_in: np.ndarray,
        train_data_out: np.ndarray,
        eval_data_in: np.ndarray,
        eval_data_out: np.ndarray,
        units_range: tuple[int, int],
        layers_range: tuple[int, int],
        activations: List[ActivationFunction],
        learning_rate_range: tuple[float, float],
        regularizers: List[str],
        lambda_range: tuple[float, float],
        momentum_range: tuple[float, float],
        stoppings: List[Early_stopping],
        epochs: List[int],
        trials: int,
        task: Optional[Literal["Classification", "Regression"]] = "Classification"
        ):
    best_eval_loss = float('inf')
    best_eval_standard_deviation = float('inf')
    best_train_loss = float('inf')
    best_eval_losses = np.array([])
    best_eval_accuracies = np.array([])
    best_nn = None
    input_size = train_data_in.shape[1]  # Number of input features
    output_size = train_data_out.shape[1] if train_data_out.ndim != 1 else 1  # Number of output features

    start_time = time.time()

    for trial in range(trials):
        config = generate_config(units_range, layers_range, activations, input_size, output_size, task)

        learning_rate_value = np.random.uniform(*learning_rate_range)
        if (random.randint(0, 1)):
            learning_rate = Learning_rate(learning_rate_value)
        else:
            learning_rate = Linear_decay_learning_rate(learning_rate_value, learning_rate_value / 2, random.randint(10, 200))

        regularization = random.choice(regularizers)
        if (regularization is not None):
            lambda_par = np.random.uniform(*lambda_range)
        else:
            lambda_par = None
        
        if (random.randint(0, 1)):
            momentum = None
        else:
            if (random.randint(0, 1)):
                momentum = Momentum(np.random.uniform(*momentum_range))
            else:
                momentum = Nesterov_momentum(np.random.uniform(*momentum_range))
        
        early_stopping = random.choice(stoppings)
        num_epochs = random.choice(epochs)

        eval_loss_values = np.array([])
        eval_losses = []
        eval_accuracies = []

        nn = FF_Neural_Network(input_size, config, learning_rate, regularization, lambda_par, momentum, early_stopping)

        for trial in range(5):
            eval_loss, eval_accuracy, train_loss = nn.train(train_data_in, train_data_out, num_epochs, False, eval_data_in, eval_data_out)
            eval_losses.append(eval_loss)
            eval_accuracies.append(eval_accuracy)
            eval_loss_values = np.append(eval_loss_values, eval_loss[-1])
            train_loss_values = np.append(train_loss_values, train_loss)
            nn.reset()

        mean_eval_loss = np.mean(eval_loss_values)
        print(f"MEAN LOSS: {mean_eval_loss}")
        mean_standard_deviation = np.std(eval_loss_values)
        mean_train_loss = np.mean(train_loss_values)
        if mean_eval_loss < best_eval_loss:
            print(f"UPDATED BEST LOSS: {mean_eval_loss}")
            best_eval_loss = mean_eval_loss
            best_eval_standard_deviation = mean_standard_deviation
            best_train_loss = mean_train_loss
            best_eval_losses = eval_losses
            best_eval_accuracies = eval_accuracies
            best_nn = nn

        if (trial % 100 == 0):
            print(f"CONFIGURATION #{trial} TRAINED")

    print(f"Best mean eval loss: {best_eval_loss}")
    print(f"Eval standard deviation: {best_eval_standard_deviation}")
    if (task == "Classification"):
        print(f"Eval accuracy: {best_eval_accuracies[-1]}")
    print(f"Best mean train loss: {best_train_loss}")
    print(f"Obtained using model: {best_nn}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    if (task == "Classification"):
        plot_cross_validation(best_eval_losses, best_eval_accuracies)
    else:
        plot_loss_cross_validation(best_eval_losses)

    return best_nn, best_train_loss


def random_search_k_fold(
        input_data: np.ndarray,
        output_data: np.ndarray,
        units_range: tuple[int, int],
        layers_range: tuple[int, int],
        activations: List[ActivationFunction],
        learning_rate_range: tuple[float, float],
        regularizers: List[str],
        lambda_range: tuple[float, float],
        momentum_range: tuple[float, float],
        stoppings: List[Early_stopping],
        epochs: List[int],
        trials: int,
        task: Optional[Literal["Classification", "Regression"]] = "Classification"
        ):
    best_eval_loss = float('inf')
    best_eval_standard_deviation = float('inf')
    best_train_loss = float('inf')
    best_eval_losses = np.array([])
    best_eval_accuracies = np.array([])
    best_nn = None
    input_size = input_data[0].shape[1]  # Number of input features
    output_size = output_data[0].shape[1] if output_data[0].ndim != 1 else 1  # Number of output features

    start_time = time.time()

    for trial in range(trials):
        config = generate_config(units_range, layers_range, activations, input_size, output_size, task)

        learning_rate_value = np.random.uniform(*learning_rate_range)
        if (random.randint(0, 1)):
            learning_rate = Learning_rate(learning_rate_value)
        else:
            learning_rate = Linear_decay_learning_rate(learning_rate_value, learning_rate_value / 2, random.randint(10, 200))

        regularization = random.choice(regularizers)
        if (regularization is not None):
            lambda_par = np.random.uniform(*lambda_range)
        else:
            lambda_par = None
        
        if (random.randint(0, 1)):
            momentum = None
        else:
            if (random.randint(0, 1)):
                momentum = Momentum(np.random.uniform(*momentum_range))
            else:
                momentum = Nesterov_momentum(np.random.uniform(*momentum_range))
        
        early_stopping = random.choice(stoppings)
        num_epochs = random.choice(epochs)

        eval_loss_values = np.array([]) #Eval losses over all folds
        all_eval_losses = []
        all_eval_accuracies = []
        train_loss_values = np.array([])
        nn = FF_Neural_Network(input_size, config, learning_rate, regularization, lambda_par, momentum, early_stopping)

        for i, fold in enumerate(input_data):
            fold_loss_values = np.array([]) #losses on different trials over a fold
            eval_losses = np.array([]) #Losses over same trials over the same fold
            eval_accuracies = np.array([])
            train_data_in = np.concatenate([input_data[j] for j in range(len(input_data)) if j != i])
            train_data_out = np.concatenate([output_data[j] for j in range(len(input_data)) if j != i])
            eval_data_in = fold
            eval_data_out = output_data[i]
            for trial in range(3):
                eval_losses, eval_accuracies, train_loss = nn.train(train_data_in, train_data_out, num_epochs, False, eval_data_in, eval_data_out)
                all_eval_losses.append(eval_losses)
                all_eval_accuracies.append(eval_accuracies)
                fold_loss_values = np.append(fold_loss_values, eval_losses[-1])
                train_loss_values = np.append(train_loss_values, train_loss)
                nn.reset()
            mean_fold_eval_loss = np.mean(fold_loss_values)
            eval_loss_values = np.append(eval_loss_values, mean_fold_eval_loss)

        mean_loss = np.mean(eval_loss_values)
        print(f"MEAN LOSS: {mean_loss}")
        mean_standard_deviation = np.std(eval_loss_values)
        mean_train_loss = np.mean(train_loss_values)
        if mean_loss < best_eval_loss:
            print(f"UPDATED BEST LOSS: {mean_loss}")
            best_eval_loss = mean_loss
            best_eval_standard_deviation = mean_standard_deviation
            best_train_loss = mean_train_loss
            best_eval_losses = eval_losses
            best_eval_accuracies = eval_accuracies
            best_nn = nn

        if (trial % 100 == 0):
            print(f"CONFIGURATION #{trial} TRAINED")

    print(f"Best mean eval loss: {best_eval_loss}")
    print(f"Eval standard deviation: {best_eval_standard_deviation}")
    if (task == "Classification"):
        print(f"Eval accuracy: {best_eval_accuracies[-1]}")
    print(f"Best mean train loss: {best_train_loss}")
    print(f"Obtained using model: {best_nn}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    if (task == "Classification"):
        plot_cross_validation(best_eval_losses, best_eval_accuracies)
    else:
        plot_loss_cross_validation(best_eval_losses)

    return best_nn, best_train_loss


def all_layer_configs(num_units: List[int], num_layers: List[int], act_funs: list[ActivationFunction], input_size: int, output_size: int, task: str):
    unit_act_configs = list()
    for i in range(len(num_layers)):
        units = list(product(num_units, repeat = num_layers[i]))
        activations = list((product(act_funs, repeat = num_layers[i])))
        prod = list(product(units, activations))
        unit_act_configs += prod

    layer_configs = []
    for config in unit_act_configs:
        layer_list = [Dense_layer(input_size, config[0][0], config[1][0])] #Input layer
        for i, num_units in enumerate(config[0]):
            if (i == len(config[0]) - 1): #output layer
                if (task == "Classification"):
                    layer1 = Dense_layer(num_units, output_size, Logistic)
                    layer2 = Dense_layer(num_units, output_size, Tanh)
                    layer_list_copy = list(layer_list)
                    layer_list.append(layer1)
                    layer_list_copy.append(layer2)
                    layer_configs.append(layer_list)
                    layer_configs.append(layer_list_copy)
                elif (task == "Regression"):
                    layer = Dense_layer(num_units, output_size, Linear)
                    layer_list.append(layer)
                    layer_configs.append(layer_list)
                else:
                    raise RuntimeError("The task must be of classification or regression")
            else: #Hidden layer
                layer = Dense_layer (num_units, config[0][i + 1], config[1][i + 1])
                layer_list.append(layer)
    
    return layer_configs

def generate_config(
        units: tuple[int, int], 
        layers: tuple[int, int], 
        activations: List[ActivationFunction],
        input_size: int,
        output_size:int,
        task:str
        ):
       
    layer_list = [Dense_layer(input_size, random.randint(*units), random.choice(activations))] #Input layer
    num_layers = random.randint(*layers)

    for i in range(num_layers):
        if (i == num_layers - 1): #Output layer
            if (task == "Classification"):
                layer = Dense_layer(layer_list[-1].num_units, output_size, random.choice([Logistic, Tanh]))
            elif (task == "Regression"):
                layer = Dense_layer(layer_list[-1].num_units, output_size, Linear)
            else:
                raise RuntimeError("The task must be of classification or regression")
        else: #Hidden layer
            layer = Dense_layer(layer_list[-1].num_units, random.randint(*units), random.choice(activations))
        layer_list.append(layer)
    
    return layer_list