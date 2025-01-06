import numpy as np
from .activations import *
from .layers import *
from .learning_rate import *
from .momentum import *
from .early_stopping import *
from .utils.metrics import *
from .utils.plot import *
from .utils.data_split import shuffle_data
from .utils.data_utils import save_predictions
from typing import List
from typing import Optional, Literal, List

class FF_Neural_Network:

    def __init__(
            self,
            input_size: int,
            layers: List[Layer],
            learning_rate: Learning_rate,
            loss: Loss,
            regularized: Optional[Literal["Lasso", "Tikhonov"]] = None,
            lambda_par: Optional[float] = None,
            momentum: Optional[Momentum] = None,
            early_stopping: Optional[Early_stopping] = None
            ):
        """
        Initializes a feedforward neural network.

        Args:
            input_size (int): The size of the input layer.
            layers (List[Layer]): List of layer objects defining the network architecture.
            learning_rate (Learning_rate): Learning rate object for weight updates.
            loss (Loss): Loss function used for training.
            regularized (Optional[Literal["Lasso", "Tikhonov"]]): Regularization type (Lasso or Tikhonov).
            lambda_par (Optional[float]): Lambda value for regularization.
            momentum (Optional[Momentum]): Momentum object for weight updates.
            early_stopping (Optional[Early_stopping]): Early stopping criterion.

        Raises:
            RuntimeError: If the input size does not match the size of the first layer.
        """

        self.input_size = input_size
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss = loss
        self.regularized = regularized
        self.lambda_par = lambda_par
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.past_grad = np.array([])

        if (self.input_size != self.layers[0].weights.shape[0]):
             raise RuntimeError("The input layer size and the input size must coincide")
    

    def fwd_computation(self, input):
        """
        Performs forward computation through the network.

        Args:
            input (np.ndarray): Input data to the network.

        Returns:
            np.ndarray: Output of the network after forward computation.
        """

        out = np.array([])
        for layer in self.layers:
            out = layer.fwd_computation(input)
            input = out
        return out
    

    def bwd_computation(self, output, pred):
        """
        Performs backpropagation to compute gradients and update weights.

        Args:
            output (np.ndarray): Ground truth labels.
            pred (np.ndarray): Predictions from the forward pass.

        Returns:
            None
        """
       
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
                    delta = self.loss.derivative(pred, output) * layer.activation.derivative(np.dot(layer.input, weights)).squeeze() 
                else:
                    delta = self.loss.derivative(pred, output) * layer.activation.derivative(layer.net).squeeze() 
               
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
                grad = (self.learning_rate() * np.dot(layer.input.T, delta)) 
                if (grad.ndim == 1):
                    grad = grad.reshape(grad.shape[0], 1)

            #Gradient clipping to avoid exploding gradient 
            grad = np.clip(grad, -0.3, 0.3)

            bias_update = self.learning_rate() * np.sum(delta, axis=0, keepdims=True)

            #Momentum
            if (isinstance(self.momentum, Momentum) and not isinstance(self.momentum, Nesterov_momentum) and self.past_grad.size != 0):
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
            test_input: Optional[np.ndarray] = None,
            test_output: Optional[np.ndarray] = None,
            best_train_loss: float = None,
            mode: Optional[Literal["Online", "Batch", "Minibatch"]] = "Batch",
            mb_number = None
            ):
        """
        Trains the neural network using the specified mode.

        Args:
            input (np.ndarray): Training input data.
            output (np.ndarray): Training output labels.
            epochs (int): Number of epochs to train the model.
            plot (bool): Whether to plot training metrics. Defaults to False.
            eval_input (Optional[np.ndarray]): Validation input data.
            eval_output (Optional[np.ndarray]): Validation output labels.
            test_input (Optional[np.ndarray]): Test input data.
            test_output (Optional[np.ndarray]): Test output labels.
            best_train_loss (float): Threshold for early stopping based on training loss.
            mode (Optional[Literal["Online", "Batch", "Minibatch"]]): Training mode (Online, Batch, Minibatch).
            mb_number (int): Number of minibatches (required if mode is Minibatch).

        Returns:
            Various training and validation metrics depending on the inputs.
        """
        
        train_losses = []
        train_accuracies = []
        eval_losses = []
        eval_accuracies = []
        test_losses = []
        test_accuracies = []

        #Adjust the outputs to -1, 1 for Tanh
        if (isinstance(self.layers[-1].activation, Tanh)):
            output = 2 * output - 1
            if (eval_output is not None):
                eval_output = 2 * eval_output - 1

        record_accuracy = True

        #We do not record accuracy if we are performing a regression task
        if (isinstance(self.layers[-1].activation, Linear)):
            record_accuracy = False

        for epoch in range(epochs):
            
            #We perform this check during retraining to ensure that the new training loss does not get lower 
            #than the one of the originally trained model, preventing overtraining and thus overfitting
            if (best_train_loss is not None and len(train_losses) != 0):
                if (train_losses[-1] < best_train_loss):
                    break

            if (output.ndim != 1): 
                predictions = np.empty((0, output.shape[1]))
            else:
                predictions = np.empty((0, 1))

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
                    predictions = np.concatenate((predictions, pred), axis=0)
                    self.bwd_computation(output_batches[i], pred)
            
            elif (mode == "Online"):
                #Shuffle input and output together to prevent sample ordering bias
                input, output = shuffle_data(input, output)
            
                for i in range(len(input)):
                    pred = self.fwd_computation(input[i])
                    predictions = np.concatenate((predictions, pred), axis=0)
                    self.bwd_computation(np.array([output[i]]), pred)
            
            else:
                raise RuntimeError(f"{mode} is not a training mode.")

            #Computation of training loss and accuracy for the current epoch
            if (record_accuracy):
                train_acc = compute_accuracy(output, predictions.reshape(predictions.shape[0]), type(self.layers[-1].activation).__name__)
                train_accuracies.append(train_acc)

            train_loss = self.loss.compute(predictions, output) #For the monk we use MSE loss
            train_losses.append(train_loss)

            #Evaluation and early stopping
            if (eval_input is not None and eval_output is not None):
                if (self.evaluate(eval_input, eval_output, eval_losses, eval_accuracies, record_accuracy)):
                    break
            
            #Testing
            if (test_input is not None and test_output is not None):
                if (record_accuracy):
                    test_loss, test_accuracy = self.test(test_input, test_output)
                    test_losses.append(test_loss)
                    test_accuracies.append(test_accuracy)
                else:
                    test_loss = self.test(test_input, test_output)
                    test_losses.append(test_loss)
            
            #Uncomment to print information during training, to get a sense of how the model is learning
            # if (epoch % 100 == 0 or epoch == epochs - 1):
            #     if (record_accuracy):
            #         print(f"Training Accuracy at epoch {epoch + 1} = {train_acc:.4f}")
            #     print(f"Training Loss at epoch: {epoch + 1} = {train_loss:.4f}")
            #     if (eval_input is not None and eval_output is not None):
            #         if (record_accuracy):
            #             print(f"Validation Accuracy at epoch {epoch + 1} = {eval_accuracies[-1]:.4f}")
            #         print(f"Validation Loss at epoch: {epoch + 1} = {eval_losses[-1]:.4f}")

        if (plot):
            if (record_accuracy):
                loss_accuracy_plot(train_losses, train_accuracies, len(train_losses))
                if (eval_input is not None and eval_output is not None):
                    loss_accuracy_plot(eval_losses, eval_accuracies, len(eval_losses))
                elif (test_input is not None and test_output is not None):
                    loss_accuracy_plot(test_losses, test_accuracies, len(test_losses))
            else:
                plot_loss(train_losses, len(train_losses))
                if (eval_input is not None and eval_output is not None):
                    plot_loss(eval_losses, len(eval_losses))
                elif (test_input is not None and test_output is not None):
                    plot_loss(test_losses, len(test_losses))
        
        if (test_input is not None and test_output is not None):
            print(f"Test loss: {test_losses[-1]}")
            if (record_accuracy):
                print(f"Test accuracy: {test_accuracies[-1]}")

        if (eval_input is not None and eval_output is not None):
            return train_losses, eval_losses, train_accuracies, eval_accuracies
        return train_losses[-1]


    def evaluate(self, eval_input: np.ndarray, eval_output: np.ndarray, eval_losses: np.ndarray, eval_accuracies: np.ndarray, record_accuracy: bool):
        """
        Evaluates the model on validation data.

        Args:
            eval_input (np.ndarray): Validation input data.
            eval_output (np.ndarray): Validation output labels.
            eval_losses (np.ndarray): List to record validation losses.
            eval_accuracies (np.ndarray): List to record validation accuracies.
            record_accuracy (bool): Whether to record accuracy metrics.

        Returns:
            bool: True if early stopping criterion is met, False otherwise.
        """

        eval_loss = 0
        if (record_accuracy):
            eval_acc = compute_accuracy(eval_output, self.fwd_computation(eval_input).reshape(eval_output.shape[0]), type(self.layers[-1].activation).__name__)
            eval_accuracies.append(eval_acc)

        eval_loss = self.loss.compute(self.fwd_computation(eval_input), eval_output)
        if (self.early_stopping is not None):
            if (self.early_stopping(eval_losses, eval_loss)):
                eval_losses.append(eval_loss)
                return True
        eval_losses.append(eval_loss)
        return False


    def test(self, input: np.ndarray, output: np.ndarray):
        """
        Tests the model on unseen data.

        Args:
            input (np.ndarray): Test input data.
            output (np.ndarray): Test output labels.

        Returns:
            float: Test loss.
            float: Test accuracy (if applicable).
        """

        #Adjust the outputs to -1, 1 for Tanh
        if (isinstance(self.layers[-1].activation, Tanh)):
            output = 2 * output - 1

        #We do not record accuracy if we are performing a regression task
        record_accuracy = True
        if (isinstance(self.layers[-1].activation, Linear)):
            record_accuracy = False

        y_test = self.fwd_computation(input)

        if (record_accuracy):
            test_accuracy = compute_accuracy(output, y_test.reshape(y_test.shape[0]), type(self.layers[-1].activation).__name__)
        
        test_loss = self.loss.compute(y_test, output)
        
        if (record_accuracy):
            return test_loss, test_accuracy
        return test_loss
    

    def blind_test_ML_cup (self, test_input: np.ndarray):
        """
        Performs predictions on test data and saves results.

        Args:
            test_input (np.ndarray): Test input data.

        Returns:
            None
        """

        test_output = self.fwd_computation(test_input)
        save_predictions(test_output, "blind test results")

    
    def adjust_learning_rate(self, batch_size_training: int, batch_size_retraining: int):
        """
        Adjusts the learning rate when retraining on different batch sizes.

        Args:
            batch_size_training (int): Original batch size used during training.
            batch_size_retraining (int): New batch size for retraining.

        Returns:
            None
        """

        if (not isinstance(self.learning_rate, Linear_decay_learning_rate)):
            self.learning_rate.eta *= (batch_size_training / batch_size_retraining)
        else:
            self.learning_rate.eta_start *= (batch_size_training / batch_size_retraining)
            self.learning_rate.eta_tau *= (batch_size_training / batch_size_retraining)


    def reset(self):
        """
        Resets the weights of the network for retraining.

        Returns:
            None
        """
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
            f"Loss: {self.loss}\n"
            f"Regularization: {self.regularized or 'None'}\n"
            f"Lambda parameter: {self.lambda_par or 'None'}\n"
            f"Momentum parameter: {self.momentum or 'None'}\n"
            f"Early stopping: {self.early_stopping or 'None'}"
        )