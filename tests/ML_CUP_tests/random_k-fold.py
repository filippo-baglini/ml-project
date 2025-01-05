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

input, output = shuffle_data(input, output)

kfold_data_in, test_data_in, kfold_data_out, test_data_out = train_test_splitter(input, output, 0.2)

input_split, output_split = k_fold_splitter(kfold_data_in, kfold_data_out, 4) 

num_units = (2, 80)  # Possible number of units for hidden layers
num_layers = (1, 4)
act_funs = [Tanh, ReLU, Leaky_ReLU, ELU]  # Hidden layer activation functions
learning_rates = (0.00001, 0.00007)
losses = [MEE()]
regularization = [None, "Tikhonov", "Lasso"]
lambda_values = (0.00001, 0.1)
momentum_values = (0.1, 0.95)
early_stopping = [Early_stopping(30, 0.001), Early_stopping(50, 0.001)]
num_epochs = [2000]

nn, best_train_loss = random_search_k_fold(input_split, output_split, num_units, num_layers, act_funs, learning_rates, losses, regularization, lambda_values, momentum_values, early_stopping, num_epochs, 5000, "Regression")
nn.reset()

x_train = np.concatenate([input_split[i] for i in range (len(input_split) - 1)])

nn.adjust_learning_rate((x_train.shape[0]), input.shape[0])
print(f"Learning rate during retraining: {nn.learning_rate}")

nn.train(kfold_data_in, kfold_data_out, 2000, True, None, None, test_data_in, test_data_out, best_train_loss)