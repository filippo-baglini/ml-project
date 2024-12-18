from .activations import *
from typing import Optional, Literal

class Layer:

    def __init__(self):
        raise NotImplementedError
    
    def initialize_weights(self, nInputs: int, nUnits: int, initialization_technique: str):

        if (initialization_technique == "Normal Xavier"):
            return np.random.randn(nInputs, nUnits) * np.sqrt(2 / (nInputs+nUnits))

        elif (initialization_technique == "Uniform Xavier"):
            return np.random.uniform(-np.sqrt(6 / (nInputs + nUnits)), np.sqrt(6 / (nInputs + nUnits)), (nInputs, nUnits))

        elif (initialization_technique == "He"):
             return np.random.randn(nInputs, nUnits) * np.sqrt(2 / nInputs)
        
        elif (initialization_technique is None):
            return np.random.uniform(-0.7, 0.7, (nInputs, nUnits))

        else:
            raise RuntimeError("Invalid weight initialization technique.")

class Dense_layer(Layer):

    def __init__(
            self, 
            nInputs: int, 
            nUnits: int, 
            activation: ActivationFunction, 
            initialization_technique: Optional[Literal["Normal Xavier", "Uniform Xavier", "He"]] = None
            ):
        
        self.weights = self.initialize_weights(nInputs, nUnits, initialization_technique)
        self.biases = self.initialize_weights(1, nUnits, None)
        self.activation = activation()
        self.input = np.array([])
        self.net = 0
    
    def fwd_computation(self, input: np.ndarray):
        self.input = input
        self.net = self.biases + np.dot(input, self.weights)
        return self.activation.fwd(self.net)
