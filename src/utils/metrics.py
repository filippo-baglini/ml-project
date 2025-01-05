import numpy as np

def compute_accuracy(y_true, y_pred, activation_function="Logistic", threshold=0.5):
    """Compute accuracy for a neural network with an arbitrary activation function in the output layer."""

    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    # Initialize predicted labels
    predicted_labels = None

    if activation_function == "Logistic":
        # Threshold predictions for binary classification
        predicted_labels = (y_pred > threshold).astype(int)
    
    elif activation_function == "Tanh":
        # Threshold predictions at 0 for binary classification
         predicted_labels = np.where(y_pred >= 0, 1, -1)
    
    elif activation_function == "Linear":
        # For regression tasks, accuracy might not be meaningful.
        raise ValueError("Accuracy is not typically defined for regression tasks.")
    
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")
    
    # Compute accuracy
    correct_predictions = np.sum(y_true == predicted_labels)
    return correct_predictions / len(y_true)

def squared_loss(y_true, y_pred):
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    return (y_true - y_pred) ** 2


class Loss:

    def __init__(self):
        raise NotImplementedError
    
    def compute(self, predictions, output):
        raise NotImplementedError
    
    def derivative(self, predictions, output):
        raise NotImplementedError
    
    def __str__(self):
        return f"{self.type}"
    
class MSE(Loss):

    def __init__(self):
        self.type = "MSE"

    def compute(self, predictions, output):
        if (output.ndim == 1):
            predictions = np.squeeze(predictions)
        return np.mean(squared_loss(output, predictions))
    
    def derivative(self, predictions, output):
        
        return (predictions - output) 

class MEE(Loss):
    
    def __init__(self):
        self.type = "MEE"
    
    def compute(self, predictions, output):
        if (output.ndim == 1):
            return np.mean(np.sqrt((output - predictions) ** 2))
        return np.mean(np.sqrt(np.sum((output - predictions) ** 2, axis = 1)))
    
    def derivative(self, predictions, output):  
        
        if output.ndim == 1:
            differences = predictions - output
            return differences / (np.sqrt(np.power(output - predictions, 2)) + 1e-12)

        else:
            differences = predictions - output
            norms = np.linalg.norm(differences, axis=-1, keepdims=True) + 1e-12  # Avoid dividing by 0
            return differences / norms
