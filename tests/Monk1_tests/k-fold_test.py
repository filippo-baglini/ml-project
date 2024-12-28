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

x_split, y_split = k_fold_splitter(x, y, 4) #should split x, y in folds


num_units = [4, 6, 7, 8]  # Possible number of units for hidden layers
num_layers = [1]
act_funs = [Logistic, Tanh, ReLU, Leaky_ReLU]  # Hidden layer activation functions
learning_rates = [Learning_rate(0.15), Learning_rate(0.02), Learning_rate(0.025), Learning_rate(0.027), Learning_rate(0.03), Linear_decay_learning_rate(0.03, 0.02, 100), Linear_decay_learning_rate(0.025, 0.02, 50)]
regularization = [None, "Lasso"]
lambda_values = [None, 0.001, 0.0001, 0.00001]
momentum_values = [None, Momentum(0.7), Momentum(0.9), Nesterov_momentum(0.7), Nesterov_momentum(0.9), Nesterov_momentum(0.95)]
early_stopping = [Early_stopping(12, 0.00001), Early_stopping(15, 0.00001)]
num_epochs = [300]

nn, best_train_loss = grid_search_k_fold(x_split, y_split, num_units, num_layers, act_funs, learning_rates, regularization, lambda_values, momentum_values, early_stopping, num_epochs)
nn.reset()

nn.train(x, y, 300, True, None, None, x_test, y_true, best_train_loss)