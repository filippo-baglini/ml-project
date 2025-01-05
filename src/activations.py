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
    def __init__(self, alfa = 1):
        self.alfa = alfa

    def fwd(self, input):
        return (1 / (1 + np.exp(-self.alfa * input)))
    
    def derivative(self, input):
        f = self.fwd(input)
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
    def __init__(self, alfa = 0.01):
        self.alfa = alfa  # Slope for negative values

    def fwd(self, input):
        return np.maximum(self.alfa * input, input)

    def derivative(self, input):
        return np.where(input > 0, 1, self.alfa)

    
class ELU(ActivationFunction):
    def __init__(self, alfa = 1):
        self.alfa = alfa  # Slope for negative values

    def fwd(self, input):
        return np.where(input > 0, input, self.alfa * (np.exp(input) - 1))

    def derivative(self, input):
        f = self.fwd(input)
        return np.where(input > 0, 1, f + self.alfa)