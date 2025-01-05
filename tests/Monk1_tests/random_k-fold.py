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
monk_1_train = os.path.join(script_dir, "../../data/Monk/monks-1.train")
monk_1_test = os.path.join(script_dir, "../../data/Monk/monks-1.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_true = read_monk_data(monk_1_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

x, y = shuffle_data(x, y)
x_split, y_split = k_fold_splitter(x, y, 4) #should split x, y in folds

num_units = (2, 8) 
num_layers = (1, 1)
act_funs = [Logistic, Tanh, ReLU, Leaky_ReLU]  
learning_rates = (0.001, 0.05)
losses = [MSE()]
regularization = [None, "Tikhonov", "Lasso"]
lambda_values = (0.0001, 0.01)
momentum_values = (0.1, 0.95)
early_stopping = [Early_stopping(8, 0.0001), Early_stopping(6, 0.0001)]
num_epochs = [300]

nn, best_train_loss = random_search_k_fold(x_split, y_split, num_units, num_layers, act_funs, learning_rates, losses, regularization, lambda_values, momentum_values, early_stopping, num_epochs, 5000)
nn.reset()

x_train = np.concatenate([x_split[i] for i in range (len(x_split) - 1)])

nn.adjust_learning_rate((x_train.shape[0]), x.shape[0])
print(f"Learning rate during retraining: {nn.learning_rate}")

nn.train(x, y, 300, True, None, None, x_test, y_true, best_train_loss)
