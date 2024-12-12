import numpy as np
from activations import *
from units import *
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
        
        for i in range(len(self.hidden_layers)):
            if (i == 0):
                for unit in self.hidden_layers[i]:
                    unit.initialize_weights(self.input_size)
            else:
                for unit in self.hidden_layers[i]:
                    unit.initialize_weights(self.hidden_layers_size[i - 1])
            
        self.output_size = output_size
        self.output_layer = [OutputUnit(self.activation_function_out) for _ in range(self.output_size)]

        for i in range (self.output_size):
            for unit in self.output_layer:
                unit.initialize_weights(self.hidden_layers_size[-1])

        #if (regularizer): 
    
    def fwd_computation(self, input):

        if (self.input_size != len(input)):
            raise RuntimeError("The input layer size and the input size must coincide")

        #Load the input in the input layer
        print("PROVA INPUT")
        for i in range(len(input)):
            self.input_layer[i].value = input[i]
            print(self.input_layer[i].value)
        
        for i in range(len(self.hidden_layers_size)):
            print(f"NET LAYER {i}")
            if (i == 0):
                for unit in self.hidden_layers[i]:
                    unit.net += unit.weights[0]
                    for j in range (1, len(unit.weights)):
                        unit.net += self.input_layer[j - 1].value * unit.weights[j]
                    print(unit.net)
            else:
                for unit in self.hidden_layers[i]:
                    unit.net += unit.weights[0]
                    for j in range (1, len(unit.weights)):
                        unit.net += self.hidden_layers[i - 1][j - 1].activation.fwd(self.hidden_layers[i - 1][j - 1].net) * unit.weights[j]
                        #unit.net += (self.hidden_layers[i - 1][j - 1].net * self.hidden_layers[i -1][j - 1].activation) * unit.weights[j]
                    print(unit.net)
            
        print ("NET OUTPUT LAYER")
        for unit in self.output_layer:
            unit.net += unit.weights[0]
            for i in range (1, len(unit.weights)):
                unit.net += self.hidden_layers[-1][i - 1].activation.fwd(self.hidden_layers[-1][i - 1].net) * unit.weights[i]
            print(unit.net)
        
        return [self.output_layer[i].activation.fwd(self.output_layer[i].net) for i in range (len(self.output_layer))]

    def backpropagation(self, output, y):
        pass