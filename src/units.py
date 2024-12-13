import numpy as np
from activations import *

class Unit:

    def __init__(self):
        raise NotImplementedError
    
    def initialize_weights(self, number_of_weights):
        self.bias = np.random.uniform(-0.7, 0.7)
        self.weights = np.random.uniform(-0.7, 0.7, number_of_weights)
    
    def compute_out(self):
        return(self.activation.fwd(self.net))
    
    def compute_derivative(self):
        return(self.activation.derivative(self.net))

class InputUnit(Unit):

    def __init__(self):
        self.value = None

class HiddenUnit(Unit):

    def __init__(self, activation: ActivationFunction):
        self.bias = None
        self.weights = None
        self.input = None
        self.net = 0
        self.activation = activation()

class OutputUnit(Unit):
    def __init__(self, activation: ActivationFunction):
        self.bias = None
        self.weights = None
        self.input = None
        self.net = 0
        self.activation = activation()