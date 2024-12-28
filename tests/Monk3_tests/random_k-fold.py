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
from src.nn import FF_Neural_Network
from src.layers import *
from src.learning_rate import *
from src.early_stopping import *
from src.utils.metrics import *
from src.activations import *
import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_3_train = os.path.join(script_dir, "../../data/Monk/monks-3.train")
monk_3_test = os.path.join(script_dir, "../../data/Monk/monks-3.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_3_train)
x_test, y_true = read_monk_data(monk_3_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

x_split, y_split = k_fold_splitter(x, y, 4) #should split x, y in folds

num_units = (2, 10)  # Possible number of units for hidden layers
num_layers = (1, 1)
act_funs = [Logistic, Tanh, ReLU, Leaky_ReLU]  # Hidden layer activation functions
learning_rates = (0.01, 0.05)
regularization = [None, "Tikhonov", "Lasso"]
lambda_values = (0.0001, 0.01)
momentum_values = (0.1, 0.95)
early_stopping = [Early_stopping(7, 0.00001), Early_stopping(5, 0.0001)]
num_epochs = [300]

nn, best_train_loss = random_search_k_fold(x_split, y_split, num_units, num_layers, act_funs, learning_rates, regularization, lambda_values, momentum_values, early_stopping, num_epochs, 10000)
nn.reset()

nn.train(x, y, 300, True, None, None, x_test, y_true, best_train_loss)
