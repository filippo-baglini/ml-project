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
            activation_function: ActivationFunction,
            regularizer: Optional[Literal["Lasso", "Tikhonov"]] = None,
            ):
        
        self.activation_function = activation_function #Activation function of all the hidden layers' units

        self.input_size = input_size #Dimension of the input layer
        self.input_layer = [InputUnit() for _ in range(self.input_size)] #Initialization of each unit in the input layer

        self.hidden_layers_size = hidden_layers #Number and dimension of each hidden layer
        self.hidden_layers = [] 
        for units in self.hidden_layers_size: #Initialization of each unit in the hidden layers
            layer = [HiddenUnit(self.activation_function) for _ in range (units)]
            self.hidden_layers.append(layer)
        
        for i in range(len(self.hidden_layers)):
            if i == 0:
                for unit in self.hidden_layers[i]:
                    unit.initialize_weights(self.input_size)
            else:
                for unit in self.hidden_layers[i]:
                    unit.initialize_weights(self.hidden_layers_size[i - 1])
            
                

        self.output_size = output_size
        self.output_layer = [OutputUnit(self.activation_function) for _ in range(self.output_size)]

        for i in range (self.output_size):
            for unit in self.output_layer:
                unit.initialize_weights(self.hidden_layers_size[-1])

        #if (regularizer): 
