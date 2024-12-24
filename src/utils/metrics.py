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
    
    elif activation_function == "Softmax":
        # For multi-class classification, take the argmax
        predicted_labels = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)  # One-hot encoded ground truth to class indices
    
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

def mean_squared_error(y_true, y_pred):
    
    if (y_true.ndim == 1):
        y_pred = np.squeeze(y_pred)
    
    mse = np.mean(squared_loss(y_true, y_pred))
    
    return mse 

def mean_euclidean_error(y_true, y_pred):
   
    if (y_true.ndim == 1):
        y_pred = np.squeeze(y_pred)
    return np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)))