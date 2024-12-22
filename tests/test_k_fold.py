import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.utils.grid_search import *
from src.utils.plot import *
from src.utils.data_split import train_val_splitter
from src.nn import FF_Neural_Network
from src.layers import *
from src.learning_rate import *
from src.early_stopping import *
from src.metrics import *
from src.activations import *
import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../data/Monk/monks-1.train")
monk_1_test = os.path.join(script_dir, "../data/Monk/monks-1.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_true = read_monk_data(monk_1_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

shuffle_data(x, y) #just to shuffle them and prevent some ordering bias
x_split, y_split = k_fold_splitter(x, y, 5) #should split x, y in folds

nn = grid_search_k_fold(x_split, y_split)
nn.reset()

nn.train(x, y, 300, plot = True)

y_test = np.array([])
for i in range(len(x_test)):
    y_test = np.append(y_test, nn.fwd_computation(x_test[i]))
accuracy = compute_accuracy(y_true, y_test, type(nn.layers[-1].activation).__name__)
prova_loss = mean_squared_error(y_true, y_test)
print(f"Test accuracy: {accuracy}")
print(f"Test loss: {prova_loss}")