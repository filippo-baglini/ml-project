import os

import keras_tuner as kt
import numpy as np
from sklearn.model_selection import KFold

from cup_hypermodel import CupHyperModel

trainset = np.loadtxt('../../data/ML-CUP24-TR.csv', delimiter=",", skiprows=1, dtype=float)
trainset = trainset[:, 1:]

X = np.array(trainset[:, :-3])
y = np.array(trainset[:, -3:])

kfold = KFold(n_splits=4, shuffle=True)
fold_n = 1

for train_index, val_index in kfold.split(X):
    print(f"Training fold {fold_n}...")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    fold_dir = os.path.join('grid_search_results', f'fold_{fold_n}')

    hypermodel = CupHyperModel(input_shape=(X_train.shape[1],))

    # Configurazione della Grid Search
    tuner = kt.GridSearch(
        hypermodel,
        objective=['mse', 'val_mse'],
        directory=fold_dir,
        project_name='cup_nn'
    )

    # Esecuzione della ricerca
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        verbose=0
    )

    # Ottieni i migliori iperparametri
    best_hps = tuner.get_best_hyperparameters()[0]

    # Stampa i valori degli iperparametri migliori
    print("Migliori iperparametri trovati:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    fold_n += 1
