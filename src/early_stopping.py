import numpy as np

class Early_stopping:

    def __init__(self, patience: int, minimum_decrease: float):
        self.patience = patience
        self.minimum_decrease = minimum_decrease
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, eval_losses, eval_loss):
        if (len(eval_losses) != 0):
        #     if (eval_losses[-1] - eval_loss < self.minimum_decrease):
        #         self.counter += 1
        #         if (self.counter == self.patience):
        #             return True
        #     else:
        #         self.counter = 0
        # return False
            if eval_loss < self.best_loss - self.minimum_decrease:
                self.best_loss = eval_loss
                self.counter = 0  # Reset counter
            else:
                self.counter += 1  # Increment counter
            return self.counter >= self.patience
        else:
            self.best_loss = eval_loss
            return False
    
    def __str__(self):
        return (
            f"Early_stopping(patience={self.patience}, "
            f"minimum_decrease={self.minimum_decrease})"
        )