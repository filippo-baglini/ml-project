import numpy as np

class ActivationFunction:
    def fwd(self):
        raise NotImplementedError
    
    def derivative(self):
        raise NotImplementedError
    
class Logistic(ActivationFunction):
    def fwd(self, input, alfa):
        return (1 / (1 + np.exp(-alfa * input)))
    
    def derivative(self, input, alfa):
        f = (1 / (1 + np.exp(-alfa * input)))
        return f * (1 - f)

class Tanh(ActivationFunction):
    def fwd(self, input):
        return(np.tanh(input))
    
    def derivative(self, input):
        return(1 - np.tanh(input) ** 2)
    
class ReLU(ActivationFunction):
    def fwd(self, input):
        return(np.max(0, input))
    
    def derivative(self, input):
        if (input > 0):
            return 1
        return 0
    
class Leaky_ReLU(ActivationFunction):
    def fwd(self, input):
        return(max(0.1 * input, input))
    
    def derivative(self, input):
        if (input > 0):
            return 1
        return 0.01
    
class Softmax(ActivationFunction):
    def fwd(self, input):
        e_x = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def derivative(self, input):
        s = Softmax.fwd(input)
    # Create a diagonal matrix of the softmax probabilities
        diag_s = np.diag(s)
    # Outer product of the softmax vector with itself
        outer_s = np.outer(s, s)
    # Jacobian is diag(s) - outer(s, s)
        return diag_s - outer_s
