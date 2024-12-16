import numpy as np
from activations import *
from layers import *
from typing import List

class FF_Neural_Network:

    def __init__(
            self,
            input_size: int,
            layers: List[Layer]
            ):

        self.input_size = input_size
        self.layers = layers

        # if (self.input_size != self.layers[0].weights.shape):
        #     raise RuntimeError("The input layer size and the input size must coincide")
    
    def fwd_computation(self, input):
        out = np.array([])
        for layer in self.layers:
            out = layer.fwd_computation(input)
            input = out
        return out
    
    def bwd_computation(self, output, pred, lambda_par = None, learning_rate = 0.02):
        delta_prev_layer = np.array([])
        prev_layer = None

        for layer in reversed(self.layers):
            
            if (prev_layer == None): # Output layer
                delta = np.subtract(output, pred) * layer.activation.derivative(layer.net)
            
            else: #Hidden layer
                if layer.input.ndim == 1:
                    layer.input = layer.input.reshape(1, layer.input.shape[0])
                delta = np.dot(delta_prev_layer, prev_layer.weights.T) * layer.activation.derivative(layer.net)

            if (lambda_par):
                layer.weights += np.subtract((learning_rate * np.dot(layer.input.T, delta)), 2 * (lambda_par * layer.weights))
            else:
                layer.weights += learning_rate * np.dot(layer.input.T, delta)
            layer.biases += learning_rate * np.sum(delta, axis=0, keepdims=True)
            
            delta_prev_layer = delta
            prev_layer = layer
    
    def train(self, input, output):
        for i in range(len(input)):
            pred = self.fwd_computation(input[i])
            self.bwd_computation(output[i], np.array(pred))


