import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf


class Utils:

    @staticmethod
    def save_predictions(y_pred, folder = "."):
        # Assicurati che `y_pred` abbia 3 colonne
        if y_pred.shape[1] != 3:
            raise ValueError(f"Le predizioni devono avere esattamente 3 colonne (out_x, out_y, out_z). Trovato: {y_pred.shape[1]}")

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
    def plot_histories(h_list):
        """
        Plotta i grafici della Training Loss (MSE) e Validation Loss (val_MSE) per ogni fold.

        Parameters:
        - h_list: lista di oggetti history, uno per ogni fold.
        """
        import matplotlib.pyplot as plt

        num_folds = len(h_list)

        # Imposta la dimensione della figura: un grafico per ogni fold
        fig, axes = plt.subplots(num_folds, 1, figsize=(12, 4 * num_folds))

        # Garantisce che `axes` sia sempre un array (anche con una sola fold)
        if num_folds == 1:
            axes = [axes]

        for i, history in enumerate(h_list):
            # Recupera la loss (MSE) dal dizionario `history`
            mse = history.history['mse']
            val_mse = history.history.get('val_mse')  # Usa .get() per rendere opzionale val_mse

            # Plot della Training Loss
            axes[i].plot(mse, label='Training MSE', color='blue')

            # Plot della Validation Loss, se disponibile
            if val_mse is not None:
                axes[i].plot(val_mse, label='Validation MSE', color='red')

            axes[i].set_title(f'Fold {i + 1} - Training and Validation MSE')
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel('MSE')
            axes[i].legend()

            # Opzionale: aggiunge una griglia per una migliore leggibilità
            axes[i].grid(True, linestyle='--', alpha=0.7)

        # Adatta il layout per evitare sovrapposizioni
        plt.tight_layout()
        plt.show()

    @staticmethod
    def euclidean_distance(y_true, y_pred):
        return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))

    @staticmethod
    def plot_history(h):
        Utils.plot_histories([h])