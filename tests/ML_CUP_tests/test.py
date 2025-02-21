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


nn = FF_Neural_Network(12, [Dense_layer(12, 60, ELU), Dense_layer(60, 40, Leaky_ReLU), Dense_layer(40, 3, Linear)], Linear_decay_learning_rate(1e-05, 5e-06, 200), MEE(), None, None, Momentum(0.95), Early_stopping(30, 0.001))

train_data_in, eval_data_in, test_data_in, train_data_out, eval_data_out, test_data_out = hold_out_splitter(input, output, 0.2, 0.2)

best_train_losses, eval_losses, _, _ = nn.train(train_data_in, train_data_out, 2000, True, eval_data_in, eval_data_out)
best_train_loss1 = best_train_losses[-1]
best_eval_loss = eval_losses[-1]
print(f"TRAIN LOSS: {best_train_loss1}")
print(f"VALIDATION LOSS: {best_eval_loss}")
nn.reset()

retrain_data_in = np.concatenate((train_data_in, eval_data_in))
retrain_data_out = np.concatenate((train_data_out, eval_data_out))

nn.adjust_learning_rate((train_data_in.shape[0]), retrain_data_in.shape[0])
print(f"Learning rate during first retraining: {nn.learning_rate}")

best_train_loss2 = nn.train(retrain_data_in, retrain_data_out, 2000, True, None, None, test_data_in, test_data_out, best_train_loss1)
nn.reset()
print(f"TRAIN LOSS: {best_train_loss2}")

nn.adjust_learning_rate(retrain_data_in.shape[0], input.shape[0])
print(f"Learning rate during final retraining: {nn.learning_rate}")

final_train_loss = nn.train(input, output, 2000, True, None, None, None, None, best_train_loss2)
print(f"Train loss of the final model for the blind test: {final_train_loss}")

#nn.blind_test_ML_cup(test_input)
