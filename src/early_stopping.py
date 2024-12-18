import numpy as np

class Early_stopping:

    def __init__(self, patience: int, minimum_decrease: float):
        self.patience = patience
        self.minimum_decrease = minimum_decrease
        self.counter = 0

    def __call__(self, eval_losses, eval_loss):
        if (len(eval_losses) != 0):
            if (eval_losses[-1] - eval_loss < self.minimum_decrease):
                self.counter += 1
                if (self.counter == self.patience):
                    return True
            else:
                self.counter = 0
        return False