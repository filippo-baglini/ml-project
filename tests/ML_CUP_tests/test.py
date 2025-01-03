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
test_data = os.path.join(script_dir, "../../data/ML_Cup/ML-CUP24-TS.csv")

input, output = readTrainingCupData(data)
input, output = shuffle_data(input, output)
test_input = readTestCupData(test_data)


nn = FF_Neural_Network(12, [Dense_layer(12, 32, ReLU),  Dense_layer(32,  3, Linear)], Learning_rate(2e-05), MEE(), None, None, Momentum(0.35), Early_stopping(30, 0.0001))

train_data_in, eval_data_in, test_data_in, train_data_out, eval_data_out, test_data_out = hold_out_splitter(input, output, 0.2, 0.2)

best_train_losses, _, _, _ = nn.train(train_data_in, train_data_out, 2000, True, eval_data_in, eval_data_out)
best_train_loss1 = best_train_losses[-1]
nn.reset()

retrain_data_in = np.concatenate((train_data_in, eval_data_in))
retrain_data_out = np.concatenate((train_data_out, eval_data_out))

nn.adjust_learning_rate((train_data_in.shape[0]), retrain_data_in.shape[0])
print(f"Learning rate during retraining: {nn.learning_rate}")

best_train_loss2 = nn.train(retrain_data_in, retrain_data_out, 2000, True, None, None, test_data_in, test_data_out, best_train_loss1)
nn.reset()

nn.adjust_learning_rate(retrain_data_in.shape[0], input.shape[0])

nn.train(input, output, 2000, True, None, None, None, None, best_train_loss2)
