import numpy as np
from activations import *
from units import *
#from criterion import MSE
from typing import Optional, Literal, List

class FF_Neural_network:

    def __init__(
            self,
            input_size: int,
            hidden_layers: List[int],
            output_size: int,
            activation_function_hl: ActivationFunction,
            activation_function_out: ActivationFunction,
            regularizer: Optional[Literal["Lasso", "Tikhonov"]] = None,
            ):
        
        self.activation_function_hl = activation_function_hl #Activation function of all the hidden layers' units
        self.activation_function_out = activation_function_out #Activation function of the output units

        self.input_size = input_size #Dimension of the input layer
        self.input_layer = [InputUnit() for _ in range(self.input_size)] #Initialization of each unit in the input layer

        self.hidden_layers_size = hidden_layers #Number and dimension of each hidden layer
        self.hidden_layers = [] 
        for units in self.hidden_layers_size: #Initialization of each unit in the hidden layers
            layer = [HiddenUnit(self.activation_function_hl) for _ in range (units)]
            self.hidden_layers.append(layer)
        
        for i in range(len(self.hidden_layers)): #Initialization of weights for each unit in each hidden layer
            if (i == 0):
                for unit in self.hidden_layers[i]:
                    unit.initialize_weights(self.input_size)
            else:
                for unit in self.hidden_layers[i]:
                    unit.initialize_weights(self.hidden_layers_size[i - 1])
            
        self.output_size = output_size
        self.output_layer = [OutputUnit(self.activation_function_out) for _ in range(self.output_size)] #Initialization of each unit in the output layer

        for i in range (self.output_size): #Initialization of weights for each unit of the output layer
            for unit in self.output_layer:
                unit.initialize_weights(self.hidden_layers_size[-1])

        #if (regularizer): 
    
    def fwd_computation(self, input):

        if (self.input_size != len(input)):
            raise RuntimeError("The input layer size and the input size must coincide")

        #Load the input in the input layer
        for i in range(len(input)):
            self.input_layer[i].value = input[i]
        
        for i in range(len(self.hidden_layers_size)):
            if (i == 0):
                for unit in self.hidden_layers[i]:
                    unit.input = [self.input_layer[j].value * unit.weights[j] for j in range(len(unit.weights))]
                    unit.net = unit.bias + np.dot(unit.input, unit.weights)
            else:
                for unit in self.hidden_layers[i]:
                    unit.input = [self.hidden_layers[i - 1][j].compute_out() for j in range (len(unit.weights))]
                    unit.net = unit.bias + np.dot(unit.input, unit.weights)
            
        for unit in self.output_layer:
            unit.input = [self.hidden_layers[-1][i].compute_out() for i in range (len(unit.weights))]
            unit.net = unit.bias + np.dot(unit.input, unit.weights)
        
        return [unit.compute_out() for unit in self.output_layer]

    def backpropagate(self, out, pred, learning_rate = 0.1):
        
        delta_out = []

        for unit in self.output_layer:
            delta_k = (out - pred) * unit.compute_derivative()
            delta_out.append(delta_k)
            unit.weights += learning_rate * (delta_k * unit.input)

        delta_prev_layer = np.array([])

        for layer_idx, layer in enumerate(reversed(self.hidden_layers)):
            delta_layer = np.array([])
            if layer_idx == 0:  # If it's the last hidden layer (first in reversed order)
                for i in range(len(layer)):
                    delta_j = np.dot(delta_out, [self.output_layer[j].weights[i] for j in range(len(self.output_layer))]) * layer[i].compute_derivative()
                    delta_layer = np.append(delta_layer, delta_j)
                    layer[i].weights += learning_rate * (delta_j * layer[i].input)
            else:  # For other hidden layers
                next_layer = self.hidden_layers[len(self.hidden_layers) - layer_idx]  # Get the next layer in the original order
                for unit in layer:
                    delta_j = np.array([np.dot(delta_prev_layer, [unit[j].weights[i] for j in range(len(next_layer))]) * unit.compute_derivative()])
                    delta_layer = np.append(delta_layer, delta_j)
                    unit.weights += learning_rate * (delta_j * unit.input)
            delta_prev_layer = delta_layer  # Update delta for the next iteration

    def train(self, input, output):
        for i in range(len(input)):
            pred = self.fwd_computation(input[i])
            self.backpropagate(output[i], np.array(pred))