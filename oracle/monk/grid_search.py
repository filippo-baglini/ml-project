import os

import keras_tuner as kt
from keras.src.backend.common.global_state import clear_session
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import KFold

from monk_hypermodel import MonkHyperModel
from monk_model import MonkModel

monk = MonkModel()

X = monk.X_train
y = monk.y_train

kfold = KFold(n_splits=4, shuffle=True, random_state=42)
fold_n = 1

print("======== Grid Search ========")
for train_index, val_index in kfold.split(X):
    print(f"Training fold {fold_n}...")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    fold_dir = os.path.join('grid_search_results', f'fold_{fold_n}')

    hypermodel = MonkHyperModel(input_shape=(X_train.shape[1],))

    # Configurazione della Grid Search
    tuner = kt.GridSearch(
        hypermodel,
        objective='accuracy',
        directory=fold_dir,
        project_name='monk_nn'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Esecuzione della ricerca
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        callbacks=[early_stopping],
        verbose=0
    )

    # Ottieni i migliori iperparametri
    best_hps = tuner.get_best_hyperparameters()[0]

    # Stampa i valori degli iperparametri migliori
    print("Migliori iperparametri trovati:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    fold_n += 1

    clear_session()
    del tuner
    print("\n")