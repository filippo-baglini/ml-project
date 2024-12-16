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
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

nn = FF_Neural_Network(17, [Dense_layer(17, 4, ReLU), Dense_layer(4, 1, Logistic)], 0.02)

losses = []
accuracies = []

for epoch in range(700):
    nn.train(x, y, 'Online')
    #print(f"Training done for epoch: {epoch}")
    y_out = np.array([])
    # print(x_test.shape)
    for j in range(len(x)):
        y_out = np.append(y_out, (nn.fwd_computation(x[j])))
        #print(y_out)
    acc = binary_accuracy(y, y_out)
    loss = mean_squared_error(y, y_out)

    losses.append(loss)
    accuracies.append(acc)
    if (epoch % 100 == 0 or epoch == 699):
        print(f"Accuracy at epoch {epoch + 1} = {acc:.4f}")
        print(f"Loss at epoch: {epoch + 1} = {loss:.4f}")

prova = np.array([])
for i in range(len(x_test)):
    prova = np.append(prova, nn.fwd_computation(x_test[i]))
accuracy = binary_accuracy(y_true, prova)
prova_loss = mean_squared_error(y_true, prova)
print(f"Test accuracy: {accuracy}")
print(f"Test loss: {prova_loss}")

provaplot(losses, accuracies, 700)