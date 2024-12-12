import nn
import activations
import units
import numpy as np

nn = nn.FF_Neural_network(3, [2, 3], 5, activations.Tanh)

for i in range(len(nn.hidden_layers)):
    for j in range(len(nn.hidden_layers[i])):
            print(nn.hidden_layers[i][j].weights)
            print(nn.hidden_layers[i][j].activation)

for i in range (nn.output_size):
      print(nn.output_layer[i].weights)
      print(nn.output_layer[i].activation)