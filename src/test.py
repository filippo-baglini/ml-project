import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from utils.data_utils import *
from utils.plot import *
from utils.train_val_split import train_val_splitter
from nn import FF_Neural_Network
from layers import *
from metrics import *
from activations import *
import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../data/monks-2.train")
monk_1_test = os.path.join(script_dir, "../data/monks-2.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_true = read_monk_data(monk_1_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

x_train, x_val, y_train, y_val = train_val_splitter(x, y, 0.3)

nn = FF_Neural_Network(17, [Dense_layer(17, 4, Tanh), Dense_layer(4, 1, Logistic)], 0.2, None, None, None, 0.0001, 100)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
early_stopping_counter = 0
early_stopping_epoch = 699
for epoch in range(700):
    nn.train(x_train, y_train, 'Online')
   
    y_out_train = nn.fwd_computation(x_train)
    y_out_val = nn.fwd_computation(x_val)
   
    y_out_train = y_out_train.reshape(y_out_train.shape[0])
    y_out_val = y_out_val.reshape(y_out_val.shape[0])

    train_acc = binary_accuracy(y_train, y_out_train)
    train_loss = mean_squared_error(y_train, y_out_train)
    val_acc = binary_accuracy(y_val, y_out_val)
    val_loss = mean_squared_error(y_val, y_out_val)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    if (nn.early_stopping_decrease):
        if (len(val_losses) != 0):
            if (val_losses[-1] - val_loss < nn.early_stopping_decrease):
                nn.early_stopping_epochs += 1
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                if (nn.early_stopping_epochs == 100):
                    print("EARLY STOPPING PARTITO BLOCCA TUTTO")
                    early_stopping_epoch = epoch
                    break
            else:
                nn.early_stopping_epochs = 0
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
        else:
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
    else:
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    if (epoch % 100 == 0 or epoch == 699):
        print(f"Training Accuracy at epoch {epoch + 1} = {train_acc:.4f}")
        print(f"Training Loss at epoch: {epoch + 1} = {train_loss:.4f}")
        print(f"Validation Accuracy at epoch {epoch + 1} = {val_acc:.4f}")
        print(f"Validation Loss at epoch: {epoch + 1} = {val_loss:.4f}")

y_test = np.array([])
for i in range(len(x_test)):
    y_test = np.append(y_test, nn.fwd_computation(x_test[i]))
accuracy = binary_accuracy(y_true, y_test)
prova_loss = mean_squared_error(y_true, y_test)
print(f"Test accuracy: {accuracy}")
print(f"Test loss: {prova_loss}")

provaplot(train_losses, train_accuracies, early_stopping_epoch + 1)
provaplot(val_losses, val_accuracies, early_stopping_epoch + 1)