import numpy as np
from .activations import *
from .layers import *
from .learning_rate import *
from .momentum import *
from .early_stopping import *
from .utils.metrics import *
from .utils.plot import *
from .utils.data_split import shuffle_data
import time
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
        self.early_stopping = early_stopping
        self.past_grad = np.array([])

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

            grad = np.clip(grad, -0.3, 0.3)

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
        
        #Adjust the outputs to -1, 1 for Tanh
        if (isinstance(self.layers[-1].activation, Tanh)):
            output = 2 * output - 1
            if (eval_output is not None):
                eval_output = 2 * eval_output - 1
        
        train_losses = []
        train_accuracies = []
        eval_losses = []
        eval_accuracies = []
        early_stopping_epoch = epochs - 1

        record_accuracy = True

        #We do not record accuracy if we are performing a regression task
        if (isinstance(self.layers[-1].activation, Linear)):
            record_accuracy = False

        for epoch in range(epochs):

            if (output.ndim != 1): 
                predictions = np.empty((0, output.shape[1]))
            else:
                predictions = np.array([])

            if (mode == 'Batch'):
                pred = self.fwd_computation(input)
                predictions = pred
                self.bwd_computation(output, pred)
            
            elif (mode == 'Minibatch'):
                if (mb_number == None):
                    raise RuntimeError("If you want to train using minibatch you need to specify the number of batches.")

                # Shuffle input and output together to prevent sample ordering bias
                input, output = shuffle_data(input, output)
                
                input_batches = np.array_split(input, mb_number)
                output_batches = np.array_split(output, mb_number)
                
                for i, batch in enumerate(input_batches):
                    pred = self.fwd_computation(batch)
                    predictions = np.append(predictions, pred)
                    self.bwd_computation(output_batches[i], pred)
            
            elif (mode == "Online"):
                #Shuffle input and output together to prevent sample ordering bias
                input, output = shuffle_data(input, output)
            
                for i in range(len(input)):
                    pred = self.fwd_computation(input[i])
                    predictions = np.append(predictions, pred)
                    self.bwd_computation(np.array([output[i]]), pred)
            
            else:
                raise RuntimeError(f"{mode} is not a training mode.")

            if (record_accuracy):
                train_acc = compute_accuracy(output, predictions.reshape(predictions.shape[0]), type(self.layers[-1].activation).__name__)
                train_accuracies.append(train_acc)

            if (isinstance(self.layers[-1].activation, Linear)):
                train_loss = mean_euclidean_error(output, predictions) #For the cup we use MEE loss
            else:
                train_loss = mean_squared_error(output, predictions) #For the monk we use MSE loss
            train_losses.append(train_loss)

            #Evaluation and early stopping
            if (eval_input is not None and eval_output is not None):
                if (self.evaluate(eval_input, eval_output, eval_losses, eval_accuracies, record_accuracy)):
                    early_stopping_epoch = epoch
                    break
            
            # if (epoch % 100 == 0 or epoch == epochs - 1):
            #     print(f"Training Accuracy at epoch {epoch + 1} = {train_acc:.4f}")
            #     print(f"Training Loss at epoch: {epoch + 1} = {train_loss:.4f}")
            #     if (eval_input is not None and eval_output is not None):
            #         print(f"Validation Accuracy at epoch {epoch + 1} = {eval_accuracies[-1]:.4f}")
            #         print(f"Validation Loss at epoch: {epoch + 1} = {eval_losses[-1]:.4f}")

        if (plot):
            if (record_accuracy):
                provaplot(train_losses, train_accuracies, early_stopping_epoch + 1)
                if (eval_input is not None and eval_output is not None):
                    provaplot(eval_losses, eval_accuracies, early_stopping_epoch + 1)
            else:
                plot_loss(train_losses, early_stopping_epoch + 1)
                if (eval_input is not None and eval_output is not None):
                    plot_loss(eval_losses, early_stopping_epoch + 1)

        if (eval_input is not None and eval_output is not None):
            return eval_losses, eval_accuracies
        return 
    
    def evaluate(self, eval_input: np.ndarray, eval_output: np.ndarray, eval_losses: np.ndarray, eval_accuracies: np.ndarray, record_accuracy: bool):

        eval_loss = 0
        if (record_accuracy):
            eval_acc = compute_accuracy(eval_output, self.fwd_computation(eval_input).reshape(eval_output.shape[0]), type(self.layers[-1].activation).__name__)
            eval_accuracies.append(eval_acc)
        if (isinstance(self.layers[-1].activation, Linear)):
            eval_loss = mean_euclidean_error(eval_output, self.fwd_computation(eval_input)) #For the cup we use MEE loss
        else:
            eval_loss = mean_squared_error(eval_output, self.fwd_computation(eval_input)) #For the monk we use MSE loss
        if (self.early_stopping is not None):
            if (self.early_stopping(eval_losses, eval_loss)):
                #print(f"Early stopping activated, halting training at epoch {epoch}.")
                eval_losses.append(eval_loss)
                return True
        eval_losses.append(eval_loss)
        return False

    def test(self, input: np.ndarray, output: np.ndarray):

        #Adjust the outputs to -1, 1 for Tanh
        if (isinstance(self.layers[-1].activation, Tanh)):
            output = 2 * output - 1

        #We do not record accuracy if we are performing a regression task
        record_accuracy = True
        if (isinstance(self.layers[-1].activation, Linear)):
            record_accuracy = False

        y_test = self.fwd_computation(input)

        if (record_accuracy):
            accuracy = compute_accuracy(output, y_test.reshape(y_test.shape[0]), type(self.layers[-1].activation).__name__)
        
        if (isinstance(self.layers[-1].activation, Linear)):
            test_loss = mean_euclidean_error(output, y_test) #For the cup we use MEE loss
        else:
            test_loss = mean_squared_error(output, y_test) #For the monk we use MSE loss
        
        if (record_accuracy):
            print(f"Test accuracy: {accuracy}")
        print(f"Test loss: {test_loss}")

    def retrain(self, train_data_in: np.ndarray, train_data_out: np.ndarray, best_eval_loss: float, epochs: int):
        #CAMBIARE
        for epoch in range(epochs):
            self.train(train_data_in, train_data_out, 1)
            train_loss = mean_squared_error(train_data_out, self.fwd_computation(train_data_in))
            print(train_loss)
            if (train_loss < best_eval_loss):
                break

    def reset(self):
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