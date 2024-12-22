import numpy as np

def binary_accuracy(y_true, y_pred):

    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    y_pred = np.round(y_pred).astype(int)
    #print(f"y_true = {y_true}, y_pred = {y_pred}")
    
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)

def compute_accuracy(y_true, y_pred, activation_function="Logistic", threshold=0.5):
    """
    Compute accuracy for a neural network with an arbitrary activation function in the output layer.

    Parameters:
        y_true: Ground truth labels (array-like).
        y_pred: Predicted values from the output layer (array-like).
        activation_function: The activation function used in the output layer 
                             ("sigmoid", "tanh", "softmax", "linear").
        threshold: Threshold for binary classification (default is 0.5, used for sigmoid and tanh).

    Returns:
        Accuracy as a float (correct_predictions / total_samples).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    # Initialize predicted labels
    predicted_labels = None

    if activation_function == "Logistic":
        # Threshold predictions for binary classification
        predicted_labels = (y_pred > threshold).astype(int)
    
    elif activation_function == "Tanh":
        # Threshold predictions at 0 for binary classification
        predicted_labels = (y_pred > 0).astype(int)  # Map Tanh to {0, 1}

        # if y_pred.ndim > 1:
        #     y_pred = np.argmax(y_pred, axis=1)
    
        # if y_true.ndim > 1:
        #     y_true = np.argmax(y_true, axis=1)
        
        # Convert y_true to {0, 1} if labels are {-1, 1}
        if np.any((y_true != 0) & (y_true != 1)):
            y_true = (y_true > 0).astype(int)
    
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

def mean_squared_error(y_true, y_pred, weights = None, regularization = None, lambda_par = None):
    
    # Compute the mean squared error
    mse = np.mean(squared_loss(y_true, y_pred))
    penalty = lambda_par * sum(w.ravel().dot(w.ravel()) for w in weights) if weights else 0.0


    # Add regularization penalty
    penalty = 0.0
    if (weights is not None and regularization is not None):
        
        if (regularization == "Tikhonov"):
            penalty = lambda_par * np.sum([np.sum(w ** 2) for w in weights])
        elif (regularization == "Lasso"):
            penalty = lambda_par * np.sum([np.sum(np.abs(w)) for w in weights])
        else:
            raise RuntimeError(f"Unsupported regularization type: {regularization}")
    
    return mse + penalty