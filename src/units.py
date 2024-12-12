import numpy as np
from activations import *

class Unit:

    def __init__(self):
        raise NotImplementedError
    
    def initialize_weights(self, number_of_weights):
        self.weights = np.random.uniform(-0.7, 0.7, (number_of_weights + 1))

class InputUnit(Unit):

    def __init__(self):
        self.value = None

class HiddenUnit(Unit):

    def __init__(self, activation: ActivationFunction):
        self.weights = None
        self.activation = activation

class OutputUnit(Unit):
    def __init__(self, activation: ActivationFunction):
        self.weights = None
        self.activation = activation