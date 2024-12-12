from nn import FF_Neural_network
from activations import *
import units
import numpy as np

nn = FF_Neural_network(3, [2, 3], 1, Tanh, Logistic)

for i in range(len(nn.hidden_layers)):
    for j in range(len(nn.hidden_layers[i])):
            print(nn.hidden_layers[i][j].weights)
            print(nn.hidden_layers[i][j].activation)

for i in range (nn.output_size):
      print(nn.output_layer[i].weights)
      print(nn.output_layer[i].activation)

out = nn.fwd_computation([2,3,4])
print(out)