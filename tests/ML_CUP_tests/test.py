import sys
import os
# Get the absolute path of the parent directory of the project
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "../.."))

# Append the project directory to sys.path
sys.path.append(project_dir)

from src.utils.data_utils import *
from src.utils.plot import *
from src.utils.data_split import *
from src.nn import FF_Neural_Network
from src.layers import *
from src.learning_rate import *
from src.momentum import *
from src.early_stopping import *
from src.utils.metrics import *
from src.activations import *
import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
data = os.path.join(script_dir, "../../data/ML_Cup/ML-CUP24-TR.csv")

input, output = readTrainingCupData(data)
print(input.shape)
print(output.shape)

nn = FF_Neural_Network(12, [Dense_layer(12, 10, Logistic, 'Normal Xavier'), Dense_layer(10, 10, ReLU, 'He'), Dense_layer(10,  3, Linear)], Learning_rate(0.00001), None, None, Nesterov_momentum(0.9))

train_data_in, eval_data_in, test_data_in, train_data_out, eval_data_out, test_data_out = hold_out_splitter(input, output, 0.25, 0.25)
# print(train_data_in.shape)
# print(eval_data_in.shape)
# print(test_data_in.shape)

nn.train(train_data_in, train_data_out, 4000, True, eval_data_in, eval_data_out)
nn.reset()

retrain_data_in = np.concatenate((train_data_in, eval_data_in))
retrain_data_out = np.concatenate((train_data_out, eval_data_out))
nn.train(retrain_data_in, retrain_data_out, 4000, True, None, None)

nn.test(test_data_in, test_data_out)