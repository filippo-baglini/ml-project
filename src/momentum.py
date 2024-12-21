import numpy as np

class Momentum:

    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value
    
    def __str__(self):
        return f"Momentum(value={self.value})"
    
class Nesterov_momentum(Momentum):
    def __init__(self, value):
        self.value = value
    
    def __call__(self):
        return self.value
    
    def __str__(self):
        return f"Nesterov_momentum(value={self.value})"