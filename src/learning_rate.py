import numpy as np

class Learning_rate:

    def __init__(self, eta):
        
        if (eta <= 0 or eta > 1):
            raise RuntimeError("The learning rate value must be in the (0, 1] range.")
        
        self.eta = eta
    
    def __call__(self):
        return self.eta
    
    def __str__(self):
        return f"Learning_rate(eta={self.eta})"
    
class Linear_decay_learning_rate(Learning_rate):

    def __init__(self, eta_start, eta_tau, tau):

        if (eta_tau >= eta_start):
            raise RuntimeError("Eta_tau must be smaller than eta_start.")

        if (eta_start <= 0 or eta_start > 1 or eta_tau <= 0 or eta_tau > 1):
            raise RuntimeError("The learning rate value must be in the (0, 1] range.")
        
        self.eta_start = eta_start
        self.eta_tau = eta_tau
        self.tau = tau
        self.counter = 0
    
    def __call__(self):

        if (self.counter == self.tau):
            return self.eta_tau

        #self.counter += 1
        eta = (1 - (self.counter / self.tau)) * self.eta_start + (self.counter / self.tau) * self.eta_tau
        return eta
    
    def update_counter(self):
        self.counter += 1
    
    def __str__(self):
        return (
            f"Linear_decay_learning_rate(eta_start={self.eta_start}, "
            f"eta_tau={self.eta_tau}, tau={self.tau})"
        )