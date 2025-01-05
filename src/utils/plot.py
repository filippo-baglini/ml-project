import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def loss_accuracy_plot(losses, accuracies, epochs):

    # Plot loss curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accuracies, label='Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.show()
    plt.close()  # Close the plot to avoid memory issues

def plot_loss(losses, epochs):
    """
    Plot the loss curve.

    Parameters:
    - losses: List or array of loss values for each epoch.
    - epochs: Number of epochs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, label='Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
    plt.close()  # Close the plot to avoid memory issues


def plot_loss_cross_validation(losses, folds=None):
    """
    Plots the loss curves for multiple runs, handling varying lengths, with separate plots for each fold.

    Parameters:
    losses (list of np.ndarray): A list where each element is an array of losses for a run.
    folds (int): Number of folds to divide the runs into. If None, plots all runs together.
    """
    num_runs = len(losses)

    if folds is not None:
        # Split the losses into folds
        fold_size = len(losses) // folds
        for fold_idx in range(folds):
            fold_losses = losses[fold_idx * fold_size:(fold_idx + 1) * fold_size]

            # Create a new figure for each fold
            plt.figure(figsize=(8, 5))

            # Plot loss curves for this fold
            for i, trial in enumerate(fold_losses):
                plt.plot(range(1, len(trial) + 1), trial, label=f'Trial {i + 1}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve - Fold {fold_idx + 1}')
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        # Default behavior: Plot all runs together
        plt.figure(figsize=(8, 5))

        # Plot loss curves for all runs
        for i, run_losses in enumerate(losses):
            plt.plot(range(1, len(run_losses) + 1), run_losses, label=f'Run {i + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.tight_layout()
        plt.show()



def plot_cross_validation(losses, accuracies, folds=None):
    """
    Plots the loss and accuracy curves for multiple runs, handling varying lengths.

    Parameters:
    losses (list of np.ndarray): A list where each element is an array of losses for a run.
    accuracies (list of np.ndarray): A list where each element is an array of accuracies for a run.
    folds (int): Number of folds to divide the runs into. If None, plots all runs together.
    """
    if len(losses) != len(accuracies):
        raise ValueError("The number of loss and accuracy runs must be the same.")

    num_runs = len(losses)

    if folds is not None:
        fold_size = len(losses) // folds
        for fold_idx in range(folds):
            fold_losses = losses[fold_idx * fold_size:(fold_idx + 1) * fold_size]
            fold_accuracies = accuracies[fold_idx * fold_size:(fold_idx + 1) * fold_size]

            # Create a new figure for each fold
            plt.figure(figsize=(12, 5))
            
            # Plot loss curve for this fold
            plt.subplot(1, 2, 1)
            for i, trial in enumerate(fold_losses):
                plt.plot(range(1, len(trial) + 1), trial, label=f'Trial {i + 1}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve - Fold {fold_idx + 1}')
            plt.legend()

            # Plot accuracy curve for this fold
            plt.subplot(1, 2, 2)
            for i, trial in enumerate(fold_accuracies):
                plt.plot(range(1, len(trial) + 1), trial, label=f'Trial {i + 1}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy Curve - Fold {fold_idx + 1}')
            plt.legend()

            plt.tight_layout()
            plt.show()
    else:
        # Default behavior: Plot all runs together
        plt.figure(figsize=(12, 5))

        # Plot loss curve
        plt.subplot(1, 2, 1)
        for i, run_losses in enumerate(losses):
            plt.plot(range(1, len(run_losses) + 1), run_losses, label=f'Run {i + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        # Plot accuracy curve
        plt.subplot(1, 2, 2)
        for i, run_accuracies in enumerate(accuracies):
            plt.plot(range(1, len(run_accuracies) + 1), run_accuracies, label=f'Run {i + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()