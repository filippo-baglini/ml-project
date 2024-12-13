import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from utils.data_utils import *
from utils.plot import *
from nn import FF_Neural_network
from criterion import *
from activations import *
import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../data/monks-2.train")
monk_1_test = os.path.join(script_dir, "../data/monks-2.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_true = read_monk_data(monk_1_test)

nn = FF_Neural_network(6, [3], 1, Tanh, Logistic)

mse = 0
acc = 0
for i in range(1000):
      nn.train(x, y)
      y_out = np.array([])
      # print(x_test.shape)
      for j in range(len(x_test)):
            y_out = np.append(y_out, (nn.fwd_computation(x_test[j])))
            #print(y_out)
      acc = binary_accuracy(y_true, y_out)
      mse = mean_squared_error(y_true, y_out)

      print(f"Accuracy at epoch {i} = {acc}")
      print(f"Loss at epoch: {i} = {mse}")
      #provaplot(mse, acc, i)
# for i in range (1000):
#     nn.fwd_computation(x[0])
