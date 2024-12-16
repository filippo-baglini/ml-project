import numpy as np
from activations import *
from layers import *
from typing import List
from typing import Optional, Literal, List

class FF_Neural_Network:

    def __init__(
            self,
            input_size: int,
            layers: List[Layer],
            learning_rate: float,
            regularized: Optional[Literal["Lasso", "Tikhonov"]] = None,
            lambda_par: Optional[float] = None,
            momentum_par: Optional[float] = None
            ):

        self.input_size = input_size
        self.layers = layers
        self.learning_rate = learning_rate
        self.regularized = regularized
        self.lambda_par = lambda_par
        self.momentum_par = momentum_par

        # if (self.input_size != self.layers[0].weights.shape):
        #     raise RuntimeError("The input layer size and the input size must coincide")
    
    def fwd_computation(self, input):
        out = np.array([])
        for layer in self.layers:
            out = layer.fwd_computation(input)
            input = out
        return out
    
    def bwd_computation(self, output, pred):
        #IMPLEMENTA AMMODINO MOMENTUM
        delta_prev_layer = np.array([])
        prev_layer = None
        past_grad = None

        for layer in reversed(self.layers):
            
            if (prev_layer == None): # Output layer
                delta = np.subtract(output, pred) * layer.activation.derivative(layer.net)
            
            else: #Hidden layer
                if layer.input.ndim == 1:
                    layer.input = layer.input.reshape(1, layer.input.shape[0])
                delta = np.dot(delta_prev_layer, prev_layer.weights.T) * layer.activation.derivative(layer.net)

            if (self.regularized == "Tikhonov"):
                regularization = 2 * self.lambda_par * layer.weights
                grad = np.subtract(self.learning_rate * np.dot(layer.input.T, delta), regularization)
            elif (self.regularized == "Lasso"):
                regularization = self.lambda_par * np.sign(layer.weights)
                grad = np.subtract(self.learning_rate * np.dot(layer.input.T, delta), regularization)
            else:
                grad = self.learning_rate * np.dot(layer.input.T, delta)
            
            if layer.weights.shape != grad.shape:
                print("Reshaping weights to match grad")
                layer.weights = np.zeros_like(grad)

            bias_update = self.learning_rate * np.sum(delta, axis=0, keepdims=True) / len(output)

            # Ensure biases have the correct shape
            if layer.biases.shape != bias_update.shape:
                print(f"Reshaping biases: layer.biases.shape={layer.biases.shape}, bias_update.shape={bias_update.shape}")
                layer.biases = np.zeros_like(bias_update)

            if (self.momentum_par):
                layer.weights += grad + self.momentum_par * ...
            else:
                layer.weights += grad / len(output)
            layer.biases += bias_update
            
            delta_prev_layer = delta
            prev_layer = layer
    
    def train(self, input, output, mode: Optional[Literal["Online", "Batch", "Minibatch"]] = "Batch", mb_number = None):

        #print(mode)
        if (mode == 'Batch'):
            #print("TRAINING BATCH")
            pred = np.zeros(output.shape)
            for i in range(len(input)):
                pred[i] = self.fwd_computation(input[i])
            self.bwd_computation(output, pred)
        
        elif (mode == 'Minibatch'):
            print("TRAINING MINIBATCH")
            if (mb_number == None):
                raise RuntimeError("If you want to train using minibatch you need to specify the number of batches.")

            # Shuffle input and output together to prevent sample ordering bias
            indices = np.arange(len(input))
            np.random.shuffle(indices)

            input = input[indices]
            output = output[indices]

            input_batches = np.array_split(input, mb_number)
            output_batches = np.array_split(output, mb_number)
            for i, batch in enumerate(input_batches):
                pred = np.zeros(batch.shape)
                for j in range(len(batch)):
                    pred[j] = self.fwd_computation(batch[j])
                print(pred)
                self.bwd_computation(output_batches[i], pred)
        
        elif (mode == "Online"):
            #Shuffle input and output together to prevent sample ordering bias
            indices = np.arange(len(input))
            np.random.shuffle(indices)

            input = input[indices]
            output = output[indices]

            for i in range(len(input)):
                pred = self.fwd_computation(input[i])
                self.bwd_computation(np.array([output[i]]), pred)
        



