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

train_data_in, eval_data_in, test_data_in, train_data_out, eval_data_out, test_data_out = hold_out_splitter(input, output, 0.25, 0.25)

num_units = (5, 30)  # Possible number of units for hidden layers
num_layers = (1, 4)
act_funs = [Logistic, Tanh, ReLU, Leaky_ReLU]  # Hidden layer activation functions
learning_rates = (0.00001, 0.00005)
regularization = [None, "Tikhonov", "Lasso"]
lambda_values = (0.0001, 0.01)
momentum_values = (0.1, 0.9)
early_stopping = [Early_stopping(50, 0.0001), Early_stopping(100, 0.0001)]
num_epochs = [1000]

nn, best_train_loss = random_search_hold_out(train_data_in, train_data_out,eval_data_in, eval_data_out, num_units, num_layers, act_funs, learning_rates, regularization, lambda_values, momentum_values, early_stopping, num_epochs, 1000, "Regression")
nn.reset()

retrain_data_in = np.concatenate((train_data_in, eval_data_in))
retrain_data_out = np.concatenate((train_data_out, eval_data_out))

nn.train(retrain_data_in, retrain_data_out, 300, True, None, None, best_train_loss)

nn.test(test_data_in, test_data_out)