import numpy as np
from .activations import *
from .layers import *
from .learning_rate import *
from .momentum import *
from .early_stopping import *
from .metrics import *
from .utils.plot import *
from .utils.data_split import shuffle_data
from typing import List
from typing import Optional, Literal, List

class FF_Neural_Network:

    def __init__(
            self,
            input_size: int,
            layers: List[Layer],
            learning_rate: Learning_rate,
            regularized: Optional[Literal["Lasso", "Tikhonov"]] = None,
            lambda_par: Optional[float] = None,
            momentum: Optional[Momentum] = None,
            early_stopping: Optional[Early_stopping] = None
            ):

        self.input_size = input_size
        self.layers = layers
        self.learning_rate = learning_rate
        self.regularized = regularized
        self.lambda_par = lambda_par
        self.momentum = momentum
        self.past_grad = np.array([])
        self.early_stopping = early_stopping

        if (self.input_size != self.layers[0].weights.shape[0]):
             raise RuntimeError("The input layer size and the input size must coincide")
    
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
            
            grad = 0

            if (pred.ndim == 2):
                pred = pred.squeeze()
            
            if (prev_layer == None): # Output layer
                #Nesterov momentum
                if (isinstance(self.momentum, Nesterov_momentum) and self.past_grad.size != 0):
                    past_grad = self.past_grad[0]
                    self.past_grad = np.delete(self.past_grad, 0)
                    weights = layer.weights + self.momentum() * past_grad
                    delta = (pred - output) * layer.activation.derivative(np.dot(layer.input, weights)).squeeze()
                else:
                    delta = (pred - output) * layer.activation.derivative(layer.net).squeeze()
               
            else: #Hidden layer
                if layer.input.ndim == 1:
                    layer.input = layer.input.reshape(1, layer.input.shape[0])
                if delta_prev_layer.ndim == 1:
                    delta_prev_layer = delta_prev_layer.reshape(delta_prev_layer.shape[0], 1)
                #Nesterov momentum
                if (isinstance(self.momentum, Nesterov_momentum) and self.past_grad.size != 0):
                    past_grad = self.past_grad[0]
                    self.past_grad = np.delete(self.past_grad, 0)
                    weights = layer.weights + self.momentum() * past_grad
                    delta = np.dot(delta_prev_layer, weights.T) * layer.activation.derivative(np.dot(layer.input, weights))
                else:
                    delta = np.dot(delta_prev_layer, prev_layer.weights.T) * layer.activation.derivative(layer.net)

            #Regularization
            if (self.regularized == "Tikhonov"):
                regularization = 2 * self.lambda_par * layer.weights
                grad = self.learning_rate() * np.dot(layer.input.T, delta)
                if (grad.ndim == 1):
                    grad = grad.reshape(grad.shape[0], 1)
                grad += regularization
            elif (self.regularized == "Lasso"):
                regularization = self.lambda_par * np.sign(layer.weights)
                grad = self.learning_rate() * np.dot(layer.input.T, delta)
                if (grad.ndim == 1):
                    grad = grad.reshape(grad.shape[0], 1)
                grad += regularization
            else:
                grad = self.learning_rate() * np.dot(layer.input.T, delta)
                if (grad.ndim == 1):
                    grad = grad.reshape(grad.shape[0], 1)

            grad = np.clip(grad, -0.5, 0.5)

            bias_update = self.learning_rate() * np.sum(delta, axis=0, keepdims=True)

            #Momentum
            if (not isinstance(self.momentum, Nesterov_momentum) and isinstance(self.momentum, Momentum) and self.past_grad.size != 0):
                past_grad = self.past_grad[0]
                self.past_grad = np.delete(self.past_grad, 0)
                layer.weights -= grad + self.momentum() * past_grad
                current_grad = np.append(current_grad, grad)

            else:
                layer.weights -= grad

            layer.biases -= bias_update
            delta_prev_layer = delta
            prev_layer = layer

        self.past_grad = current_grad
    
    def train(
            self,
            input: np.ndarray,
            output: np.ndarray, 
            epochs: int, 
            plot: bool = False,
            eval_input: Optional[np.ndarray] = None,
            eval_output: Optional[np.ndarray] = None,
            mode: Optional[Literal["Online", "Batch", "Minibatch"]] = "Batch",
            mb_number = None
            ):
        
        train_losses = []
        train_accuracies = []
        eval_losses = []
        eval_accuracies = []
        early_stopping_epoch = epochs - 1

        for epoch in range(epochs):
            predictions = np.array([])

            if (mode == 'Batch'):
                pred = self.fwd_computation(input)
                predictions = pred
                self.bwd_computation(output, pred)
            
            elif (mode == 'Minibatch'):
                if (mb_number == None):
                    raise RuntimeError("If you want to train using minibatch you need to specify the number of batches.")

                # Shuffle input and output together to prevent sample ordering bias
                shuffle_data(input, output)
                
                input_batches = np.array_split(input, mb_number)
                output_batches = np.array_split(output, mb_number)
                
                for i, batch in enumerate(input_batches):
                    pred = self.fwd_computation(batch)
                    predictions = np.append(predictions, pred)
                    self.bwd_computation(output_batches[i], pred)
            
            elif (mode == "Online"):
                #Shuffle input and output together to prevent sample ordering bias
                shuffle_data(input, output)
            
                for i in range(len(input)):
                    pred = self.fwd_computation(input[i])
                    predictions = np.append(predictions, pred)
                    self.bwd_computation(np.array([output[i]]), pred)
            
            else:
                raise RuntimeError(f"{mode} is not a training mode.")

            train_acc = compute_accuracy(output, predictions.reshape(predictions.shape[0]), type(self.layers[-1].activation).__name__)
            if self.regularized:
                weights = weights = [layer.weights for layer in self.layers]
                train_loss = mean_squared_error(output, predictions.reshape(predictions.shape[0]), weights, self.regularized, self.lambda_par)
            else:
                train_loss =  mean_squared_error(output, predictions.reshape(predictions.shape[0]))
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            #Evaluation and early stopping
            if (eval_input is not None and eval_output is not None):
                eval_loss = 0
                eval_acc = compute_accuracy(eval_output, self.fwd_computation(eval_input).reshape(eval_output.shape[0]), type(self.layers[-1].activation).__name__)
                if (self.regularized):
                    weights = weights = [layer.weights for layer in self.layers]
                    eval_loss = mean_squared_error(eval_output, self.fwd_computation(eval_input).reshape(eval_output.shape[0]), weights, self.regularized, self.lambda_par)
                else:
                    eval_loss = mean_squared_error(eval_output, self.fwd_computation(eval_input).reshape(eval_output.shape[0]))
                if (self.early_stopping is not None):
                    if (self.early_stopping(eval_losses, eval_loss)):
                        #print(f"Early stopping activated, halting training at epoch {epoch}.")
                        early_stopping_epoch = epoch
                        eval_losses.append(eval_loss)
                        eval_accuracies.append(eval_acc)
                        break
                eval_losses.append(eval_loss)
                eval_accuracies.append(eval_acc)
            
            # if (epoch % 100 == 0 or epoch == epochs - 1):
            #     print(f"Training Accuracy at epoch {epoch + 1} = {train_acc:.4f}")
            #     print(f"Training Loss at epoch: {epoch + 1} = {train_loss:.4f}")
            #     if (eval_input is not None and eval_output is not None):
            #         print(f"Validation Accuracy at epoch {epoch + 1} = {eval_acc:.4f}")
            #         print(f"Validation Loss at epoch: {epoch + 1} = {eval_loss:.4f}")
        
        if (plot):
            provaplot(train_losses, train_accuracies, early_stopping_epoch + 1)
            if (eval_input is not None and eval_output is not None):
                provaplot(eval_losses, eval_accuracies, early_stopping_epoch + 1)

        if (eval_input is not None and eval_output is not None):
            return eval_losses, eval_accuracies
        return 
    
    def reset (self):
        """Method to allow resetting the weights to retrain a model"""

        for layer in self.layers:
            layer.weights = layer.initialize_weights(layer.num_inputs, layer.num_units, layer.initialization_technique)
    
    def __str__(self):
        layer_descriptions = "\n".join(
            [f"Layer {i + 1}: {layer}" for i, layer in enumerate(self.layers)]
        )
        return (
            f"FF_Neural_Network:\n"
            f"Input size: {self.input_size}\n"
            f"Layers:\n{layer_descriptions}\n"
            f"Learning rate: {self.learning_rate}\n"
            f"Regularization: {self.regularized or 'None'}\n"
            f"Lambda parameter: {self.lambda_par or 'None'}\n"
            f"Momentum parameter: {self.momentum or 'None'}\n"
            f"Early stopping: {self.early_stopping or 'None'}"
        )