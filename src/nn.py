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
            momentum_par: Optional[float] = None,
            early_stopping_decrease: Optional[float] = None,
            early_stopping_epochs: Optional[int] = None
            ):

        self.input_size = input_size
        self.layers = layers
        self.learning_rate = learning_rate
        self.regularized = regularized
        self.lambda_par = lambda_par
        self.momentum_par = momentum_par
        self.past_grad = np.array([])
        self.early_stopping_decrease = early_stopping_decrease
        self.early_stopping_epochs = early_stopping_epochs

        # if (self.input_size != self.layers[0].weights.shape):
        #     raise RuntimeError("The input layer size and the input size must coincide")
    
    def fwd_computation(self, input):
        out = np.array([])
        for layer in self.layers:
            out = layer.fwd_computation(input)
            input = out
        return out
    
    def bwd_computation(self, output, pred):
       
        delta_prev_layer = np.array([])
        prev_layer = None
        current_grad = np.array([])

        for layer in reversed(self.layers):
            
            if (prev_layer == None): # Output layer
                delta = np.subtract(output, pred) * layer.activation.derivative(layer.net)
                #delta = np.sum(delta) / len(output)
            
            else: #Hidden layer
                if layer.input.ndim == 1:
                    layer.input = layer.input.reshape(1, layer.input.shape[0])
                delta = np.dot(delta_prev_layer, prev_layer.weights.T) * layer.activation.derivative(layer.net)
                #delta = np.sum(delta) / len(output)

            #Regularization
            if (self.regularized == "Tikhonov"):
                regularization = 2 * self.lambda_par * layer.weights
                grad = np.subtract(self.learning_rate * np.dot(layer.input.T, delta), regularization)
                #grad = np.sum(grad)
            elif (self.regularized == "Lasso"):
                regularization = self.lambda_par * np.sign(layer.weights)
                grad = np.subtract(self.learning_rate * np.dot(layer.input.T, delta), regularization)
                #grad = np.sum(grad)
            else:
                grad = self.learning_rate * np.dot(layer.input.T, delta)
                #grad = np.sum(grad)

            bias_update = self.learning_rate * np.sum(delta, axis=0, keepdims=True) / len(output)

            #Momentum
            if (self.momentum_par and self.past_grad.size != 0):
                past_grad = self.past_grad[0]
                self.past_grad = np.delete(self.past_grad, 0)
                layer.weights += grad + self.momentum_par * past_grad
                current_grad = np.append(current_grad, grad)
            else:
                layer.weights += grad / len(output)
            layer.biases += bias_update

            delta_prev_layer = delta
            prev_layer = layer
        self.past_grad = current_grad
    
    def train(self, input, output, mode: Optional[Literal["Online", "Batch", "Minibatch"]] = "Batch", mb_number = None):

        if (mode == 'Batch'):

            pred = self.fwd_computation(input)
            #print(pred)
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
                pred = self.fwd_computation(batch)
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
        
        else:
            raise RuntimeError(f"{mode} is not a training mode.")