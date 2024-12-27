import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(train_losses, test_losses, accuracies):
    """
    Funzione per tracciare la learning curve basata su training loss, test loss e accuratezza.
    """
    plt.figure(figsize=(12, 6))

    # Subplot per la Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve - Loss')
    plt.legend()

    # Subplot per l'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(len(accuracies)), accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve - Accuracy')
    plt.legend()

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

def plot_data_error(trError:np.ndarray, valError:np.ndarray, firstName:str, secondName:str):
    # import matplotlib.pyplot as plt
    plt.plot(trError, label=firstName)
    plt.plot(valError, label=secondName, linestyle='dashdot', color='r')
    plt.legend()
    plt.show()

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

# def plot_loss_cross_validation(losses, folds = None):

#     num_runs = len(losses)

#     # Set up plot
#     plt.figure(figsize=(12, 5))

#     # Plot loss curve
#     plt.subplot(1, 2, 1)
#     if (folds is not None):
#         losses_per_fold = np.array_split(losses, folds)
#         for fold in losses_per_fold:
#             for i, trial in enumerate(fold):
#                 plt.plot(range(1, len(trial) + 1), trial, label=f'Trial {i + 1}')
#     for i, run_losses in enumerate(losses):
#         plt.plot(range(1, len(run_losses) + 1), run_losses, label=f'Run {i + 1}')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Loss Curve')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()
#     plt.close()  # Close the plot to free memory

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