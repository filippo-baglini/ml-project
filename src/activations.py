import numpy as np

class ActivationFunction:
    def fwd(self):
        raise NotImplementedError
    
    def derivative(self):
        raise NotImplementedError
    
class Linear(ActivationFunction):
    def fwd(self, input):
        return input
    
    def derivative(self, input):
        return np.ones_like(input)
    
class Logistic(ActivationFunction):
    def fwd(self, input, alfa = 1):
        return (1 / (1 + np.exp(-alfa * input)))
    
    def derivative(self, input, alfa = 1):
        f = (1 / (1 + np.exp(-alfa * input)))
        return f * (1 - f)

class Tanh(ActivationFunction):
    def fwd(self, input):
        return(np.tanh(input))
    
    def derivative(self, input):
        return(1 - np.tanh(input) ** 2)
    
class ReLU(ActivationFunction):
    def fwd(self, input):
        return(np.maximum(0, input))
    
    def derivative(self, input):
        return np.where(input > 0, 1, 0)
    
class Leaky_ReLU(ActivationFunction):
    def fwd(self, input):
        return(np.maximum(0.01 * input, input))
    
    def derivative(self, input):
        return np.where(input > 0, 1, 0.01)
    
class Softmax(ActivationFunction):
    def fwd(self, input):
        e_x = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def derivative(self, input):
        s = self.fwd(input)
    # Create a diagonal matrix of the softmax probabilities
        diag_s = np.diag(s)
    # Outer product of the softmax vector with itself
        outer_s = np.outer(s, s)
    # Jacobian is diag(s) - outer(s, s)
        return diag_s - outer_s