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
from src.utils.data_split import *
from src.nn import FF_Neural_Network
from src.layers import *
from src.learning_rate import *
from src.early_stopping import *
from src.utils.metrics import *
from src.activations import *
import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
data = os.path.join(script_dir, "../../data/ML_Cup/ML-CUP24-TR.csv")



input, output = readTrainingCupData(data)

kfold_data_in, test_data_in, kfold_data_out, test_data_out = train_test_splitter(input, output, 0.2)

input_split, output_split = k_fold_splitter(kfold_data_in, kfold_data_out, 4) #should split x, y in folds

num_units = [10]  # Possible number of units for hidden layers
num_layers = [2, 3]
act_funs = [Logistic, Tanh]  # Hidden layer activation functions
learning_rates = [Learning_rate(0.00001), Learning_rate(0.00002), Linear_decay_learning_rate(0.00003, 0.00001, 200), Learning_rate(0.00003)]
regularization = [None, "Tikhonov"]
lambda_values = [None, 0.0001, 0.001]
momentum_values = [None, Momentum(0.9), Nesterov_momentum(0.9)]
early_stopping = [Early_stopping(10, 0.0001)]
num_epochs = [1000]

nn, best_train_loss = grid_search_k_fold(input_split, output_split, num_units, num_layers, act_funs, learning_rates, regularization, lambda_values, momentum_values, early_stopping, num_epochs, task = "Regression")
nn.reset()

nn.train(kfold_data_in, kfold_data_out, 1000, True, None, None, best_train_loss)

nn.test(test_data_in, test_data_out)