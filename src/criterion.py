import numpy as np
# class MSE:

#     def __init__(self):
        
#         self.loss = 0
    
#     def mse(self, out, pred):

#         self.loss += (out -pred) ** 2
    
#     def compute_loss(self):

#         return (self.loss / self.counter)

def binary_accuracy(y_true, y_pred):
    # print(y_true.shape)
    # print(y_pred.shape)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    y_pred = np.round(y_pred).astype(int)
    
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)

def squared_loss(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    return (y_true - y_pred) ** 2

def mean_squared_error(y_true, y_pred):
    return np.mean(squared_loss(y_true, y_pred))
