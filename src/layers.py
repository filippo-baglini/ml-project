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
        
        elif (initialization_technique == "Random"):
            return np.random.uniform(-0.5, 0.5, (nInputs, nUnits))

        else:
            raise RuntimeError("Invalid weight initialization technique.")

class Dense_layer(Layer):

    def __init__(
            self, 
            nInputs: int, 
            nUnits: int, 
            activation: ActivationFunction, 
            initialization_technique: Optional[Literal["Normal Xavier", "Uniform Xavier", "He", "Random"]] = None
            ):
        
        self.num_inputs = nInputs
        self.num_units = nUnits
        self.activation = activation()
        
        if (initialization_technique is None and (isinstance(self.activation, Tanh) or isinstance(self.activation, Logistic))):
            initialization_technique = "Normal Xavier"
        elif (initialization_technique is None and (isinstance(self.activation, ReLU) or isinstance(self.activation, Leaky_ReLU) or isinstance(self.activation, ELU))):
            initialization_technique = "He"
        elif (initialization_technique is None):
            initialization_technique = "Random"

        self.initialization_technique = initialization_technique
        self.weights = self.initialize_weights(nInputs, nUnits, initialization_technique)
        self.biases = self.initialize_weights(1, nUnits, "Random")
        self.input = np.array([])
        self.net = 0
    

    def fwd_computation(self, input: np.ndarray):
        self.input = input
        self.net = self.biases + np.dot(input, self.weights)
        return self.activation.fwd(self.net)
    

    def __str__(self):
        return (
            f"Dense_layer(nInputs={self.weights.shape[0]}, "
            f"nUnits={self.weights.shape[1]}, "
            f"activation={self.activation.__class__.__name__}, "
            f"initialization_technique={self.initialization_technique})"
        )
