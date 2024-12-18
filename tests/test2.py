import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.utils.plot import *
from src.utils.train_val_split import train_val_splitter
from src.nn import FF_Neural_Network
from src.layers import *
from src.learning_rate import *
from src.early_stopping import *
from src.metrics import *
from src.activations import *
import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../data/Monk/monks-2.train")
monk_1_test = os.path.join(script_dir, "../data/Monk/monks-2.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_true = read_monk_data(monk_1_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

x_train, x_val, y_train, y_val = train_val_splitter(x, y, 0.3)

nn = FF_Neural_Network(17, [Dense_layer(17, 8, Tanh), Dense_layer(8, 1, Logistic)], Learning_rate(0.05), None, None, None, Early_stopping(100, 0.0001))

nn.train(x_train, y_train, 700, x_val, y_val)

y_test = np.array([])
for i in range(len(x_test)):
    y_test = np.append(y_test, nn.fwd_computation(x_test[i]))
accuracy = compute_accuracy(y_true, y_test)
prova_loss = mean_squared_error(y_true, y_test)
print(f"Test accuracy: {accuracy}")
print(f"Test loss: {prova_loss}")