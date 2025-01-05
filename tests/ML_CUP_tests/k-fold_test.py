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
test_data = os.path.join(script_dir, "../../data/ML_Cup/ML-CUP24-TS.csv")

input, output = readTrainingCupData(data)
input, output = shuffle_data(input, output)
test_input = readTestCupData(test_data)

kfold_data_in, test_data_in, kfold_data_out, test_data_out = train_test_splitter(input, output, 0.2)

input_split, output_split = k_fold_splitter(kfold_data_in, kfold_data_out, 4) #should split x, y in folds

num_units = [15, 35, 40, 50, 60, 75, 80]  # Possible number of units for hidden layers
num_layers = [1, 2]
act_funs = [ELU, Leaky_ReLU]  # Hidden layer activation functions
learning_rates = [Linear_decay_learning_rate(0.00001, 0.000005, 200), Linear_decay_learning_rate(0.000025, 0.00001, 50), Linear_decay_learning_rate(0.00003, 0.000015, 150)]
losses = [MEE()]
regularization = [None, "Tikhonov"]
lambda_values = [None, 0.0001, 0.00001]
momentum_values = [None, Momentum(0.35), Momentum(0.45), Momentum(0.5), Momentum(0.9), Momentum(0.95)]
early_stopping = [Early_stopping(30, 0.001)]
num_epochs = [2000]

nn, best_train_loss1 = grid_search_k_fold(input_split, output_split, num_units, num_layers, act_funs, learning_rates, losses, regularization, lambda_values, momentum_values, early_stopping, num_epochs, task = "Regression")
nn.reset()

x_train = np.concatenate([input_split[i] for i in range (len(input_split) - 1)])
x_train_eval = np.concatenate([input_split[i] for i in range (len(input_split))])

nn.adjust_learning_rate((x_train.shape[0]), x_train_eval.shape[0])
print(f"Learning rate during first retraining: {nn.learning_rate}")

best_train_loss2 = nn.train(kfold_data_in, kfold_data_out, 2000, True, None, None, test_data_in, test_data_out, best_train_loss1)

nn.reset()

nn.adjust_learning_rate((x_train_eval.shape[0]), input.shape[0])
print(f"Learning rate during final retraining: {nn.learning_rate}")

train_loss = nn.train(input, output, 2000, True, None, None, None, None, best_train_loss2)
print(f"Train loss of the final model for the blind test: {train_loss}")
time.sleep(10)
nn.blind_test_ML_cup(test_input)

#Learning_rate(0.000015), Learning_rate(0.00002), Linear_decay_learning_rate(0.0000225, 0.00001, 50), Linear_decay_learning_rate(0.00005, 0.000025, 60), Linear_decay_learning_rate(0.00006, 0.00003, 140), Linear_decay_learning_rate(0.000045, 0.00002, 20), Linear_decay_learning_rate(0.00001, 0.000005, 200), Linear_decay_learning_rate(0.000025, 0.00001, 50), Linear_decay_learning_rate(0.00003, 0.000015, 150)