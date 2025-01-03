import sys
import os
# Get the absolute path of the parent directory of the project
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "../.."))

# Append the project directory to sys.path
sys.path.append(project_dir)

from src.utils.data_utils import *
from src.utils.grid_search import *
from src.utils.plot import *
from src.utils.data_split import train_val_splitter
from src.nn import FF_Neural_Network
from src.layers import *
from src.learning_rate import *
from src.early_stopping import *
from src.utils.metrics import *
from src.activations import *
import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../../data/Monk/monks-1.train")
monk_1_test = os.path.join(script_dir, "../../data/Monk/monks-1.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_true = read_monk_data(monk_1_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

x, y = shuffle_data(x, y)
x_split, y_split = k_fold_splitter(x, y, 4) #should split x, y in folds


num_units = [4, 5, 6, 7]  # Possible number of units for hidden layers
num_layers = [1]
act_funs = [Logistic, Tanh, ReLU, Leaky_ReLU]  # Hidden layer activation functions
learning_rates = [Learning_rate(0.02), Learning_rate(0.022), Learning_rate(0.025), Learning_rate(0.03), Linear_decay_learning_rate(0.027, 0.02, 100), Linear_decay_learning_rate(0.03, 0.015, 100)]
losses = [MSE()]
regularization = [None]
lambda_values = [None]
momentum_values = [None, Momentum(0.2), Momentum(0.5), Momentum(0.9), Nesterov_momentum(0.2), Nesterov_momentum(0.5), Nesterov_momentum(0.9)]
early_stopping = [Early_stopping(6, 0.0001), Early_stopping(8, 0.0001)]
num_epochs = [300]


nn, best_train_loss = grid_search_k_fold(x_split, y_split, num_units, num_layers, act_funs, learning_rates, losses, regularization, lambda_values, momentum_values, early_stopping, num_epochs)
nn.reset()

x_train = np.concatenate([x_split[i] for i in range (len(x_split) - 1)])

nn.adjust_learning_rate((x_train.shape[0]), x.shape[0])
print(f"Learning rate during retraining: {nn.learning_rate}")

nn.train(x, y, 300, True, None, None, x_test, y_true, best_train_loss)