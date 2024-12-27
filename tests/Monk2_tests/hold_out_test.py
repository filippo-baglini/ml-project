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
monk_2_train = os.path.join(script_dir, "../../data/Monk/monks-2.train")
monk_2_test = os.path.join(script_dir, "../../data/Monk/monks-2.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_2_train)
x_test, y_true = read_monk_data(monk_2_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

x_train, x_eval, y_train, y_eval = train_val_splitter(x, y, 0.25)

num_units = [2, 4, 6]  # Possible number of units for hidden layers
num_layers = [1]
act_funs = [Logistic, Tanh, Leaky_ReLU, ReLU]  # Hidden layer activation functions
learning_rates = [Learning_rate(0.05), Learning_rate(0.04), Learning_rate(0.02)]
regularization = [None, "Tikhonov"]
lambda_values = [None, 0.0001, 0.001]
momentum_values = [None, Momentum(0.9)]
early_stopping = [Early_stopping(6, 0.00001)]
num_epochs = [300]

nn, best_train_loss = grid_search_hold_out(x_train, y_train, x_eval, y_eval, num_units, num_layers, act_funs, learning_rates, regularization, lambda_values, momentum_values, early_stopping, num_epochs)
nn.reset()

nn.train(x, y, 300, True, None, None, x_test, y_true, best_train_loss)
