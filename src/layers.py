from activations import *

class Layer:

    def __init__(self):
        raise NotImplementedError
    
    def initialize_weights(self, nInputs, nUnits):
        return np.random.uniform(-0.7, 0.7, (nInputs, nUnits))

class Dense_layer(Layer):

    def __init__(self, nInputs, nUnits, activation: ActivationFunction):
        
        self.weights = self.initialize_weights(nInputs, nUnits)
        # print(self.weights.shape)
        # print(self.weights)
        self.biases = np.random.uniform(-0.7, 0.7, (1, nUnits))
        self.activation = activation()
        self.input = np.array([])
        self.net = 0
    
    def fwd_computation(self, input):
        self.input = input
        self.net = self.biases + np.dot(input, self.weights)
        return self.activation.fwd(self.net)
