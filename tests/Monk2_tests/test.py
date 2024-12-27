import sys
import os
# Get the absolute path of the parent directory of the project
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "../.."))

# Append the project directory to sys.path
sys.path.append(project_dir)

from src.utils.data_utils import *
from src.utils.plot import *
from src.utils.data_split import train_val_splitter
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
monk_2_train = os.path.join(script_dir, "../../data/Monk/monks-2.train")
monk_2_test = os.path.join(script_dir, "../../data/Monk/monks-2.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_2_train)
#y = (2 * y - 1)
x_test, y_true = read_monk_data(monk_2_test)
#y_true = (2 * y_true - 1)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

x_train, x_eval, y_train, y_eval = train_val_splitter(x, y, 0.25)

nn = FF_Neural_Network(17, [Dense_layer(17, 6, Leaky_ReLU),  Dense_layer(6, 1, Logistic)], Learning_rate(0.01), None, None, None, Early_stopping(15, 0.00001))

_, _, best_train_loss = nn.train(x_train, y_train, 300, True, x_eval, y_eval)
nn.reset()

nn.train(x, y, 300, True, None, None, x_test, y_true, best_train_loss)
