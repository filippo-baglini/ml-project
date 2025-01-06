import math
import os

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


class Utils:

    @staticmethod
    def save_predictions(y_pred, folder="."):
        # Assicurati che `y_pred` abbia 3 colonne
        if y_pred.shape[1] != 3:
            raise ValueError(
                f"Le predizioni devono avere esattamente 3 colonne (out_x, out_y, out_z). Trovato: {y_pred.shape[1]}")

        # Prepara gli ID e combina i dati
        ids = np.arange(1, y_pred.shape[0] + 1).reshape(-1, 1)  # ID da 1 a n interi
        results = np.hstack((ids, y_pred))

        # Prepara l'header personalizzato
        header = [
            "# Benedetti Gabriele, Filippo Baglini",
            "# Life Is Life",
            "# ML-CUP24 v1",
            "# 22 Jan 2024"
        ]

        # Nome del file di output
        output_file = f"{folder}/life-is-life_ML-CUP24-TS.csv"

        # Scrive il file CSV con l'header personalizzato
        with open(output_file, "w") as f:
            f.write("\n".join(header) + "\n")
            np.savetxt(f, results, fmt="%d,%.10f,%.10f,%.10f", delimiter=",", newline="\n")

        return os.path.abspath(output_file)

    @staticmethod
    def plot_histories(histories, metrics, config=None):
        """
        Plot training metrics from multiple keras History objects.

        Args:
            histories (list): List of keras History objects
            metrics (list): List of metrics to plot (e.g. ['loss', 'accuracy'])
            config (dict): Configuration dictionary with the following optional parameters:
                - aggregate_loss (bool): If True, training and validation metrics are plotted
                                       in the same graph (default: True)
                - titles (list): List of titles for each plot. Use %i to include the fold number
                                (default: metric names)
                - split (bool): If True, creates separate plots, else combines all in one figure
                               (default: False)
        """
        # Default configuration
        default_config = {
            'aggregate_loss': True,
            'titles': None,
            'split': False
        }

        # Update default config with provided config
        if config is None:
            config = {}
        config = {**default_config, **config}

        # Generate default titles if not provided
        if config['titles'] is None:
            config['titles'] = metrics

        # Calculate number of rows and columns for subplot layout
        n_metrics = len(metrics)
        n_histories = len(histories)
        total_plots = n_metrics * n_histories

        # Determine number of columns based on metrics and configuration
        n_cols = 1  # Default to 1 column
        if not config['split'] and n_metrics > 1:
            n_cols = 2  # Use 2 columns only if we have multiple metrics

        n_rows = math.ceil(total_plots / n_cols)

        # If not split, create the figure with all subplots
        if not config['split']:
            plt.figure(figsize=(7.5 * n_cols, 5 * n_rows))

        # Plot each history
        for h_idx, history in enumerate(histories):
            for m_idx, metric in enumerate(metrics):
                if config['split']:
                    plt.figure(figsize=(10, 6))
                else:
                    plt.subplot(n_rows, n_cols, h_idx * len(metrics) + m_idx + 1)

                # Get metric data
                train_metric = history.history[metric]
                val_metric = history.history.get(f'val_{metric}')

                # Plot training metric
                plt.plot(train_metric, 'b-', label=f'Training {metric}')

                # Plot validation metric if available
                if val_metric and config['aggregate_loss']:
                    plt.plot(val_metric, 'r-', label=f'Validation {metric}')

                # Set title
                current_title = config['titles'][m_idx].replace('%i', str(h_idx + 1))
                plt.title(f'{current_title}')

                # Set labels
                plt.xlabel('Epochs')
                plt.ylabel(metric.capitalize())

                plt.legend(prop={'size': 15})
                plt.grid(True)

                # Show plot if split
                if config['split']:
                    plt.tight_layout()
                    plt.show()

        # Show combined plot if not split
        if not config['split']:
            plt.tight_layout()
            plt.show()


    @staticmethod
    def euclidean_distance(y_true, y_pred):
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))


    @staticmethod
    def plot_history(h, metrics, config=None):
        Utils.plot_histories([h], metrics, config)
