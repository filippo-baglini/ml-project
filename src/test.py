import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from utils.data_utils import *
from utils.plot import *
from nn import FF_Neural_Network
from layers import *
from metrics import *
from activations import *
import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../data/monks-1.train")
monk_1_test = os.path.join(script_dir, "../data/monks-1.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_true = read_monk_data(monk_1_test)

nn = FF_Neural_Network(6, [Dense_layer(6, 5, Tanh), Dense_layer(5, 1, Logistic)])

losses = []
accuracies = []

for epoch in range(1000):
    nn.train(x, y)
    #print(f"Training done for epoch: {epoch}")
    y_out = np.array([])
    # print(x_test.shape)
    for j in range(len(x_test)):
        y_out = np.append(y_out, (nn.fwd_computation(x_test[j])))
        #print(y_out)
    acc = binary_accuracy(y_true, y_out)
    loss = mean_squared_error(y_true, y_out)

    losses.append(loss)
    accuracies.append(acc)

    print(f"Accuracy at epoch {epoch + 1} = {acc}")
    print(f"Loss at epoch: {epoch + 1} = {loss}")

provaplot(losses, accuracies, 1000)