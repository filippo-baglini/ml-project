import numpy as np
import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


def plot_histories(h_list):
    """
    Plotta i grafici della Training Loss (MSE) e Validation Loss (val_MSE) per ogni fold.

    Parameters:
    - h_list: lista di oggetti history, uno per ogni fold.
    """
    num_folds = len(h_list)

    # Imposta la dimensione della figura: un grafico per ogni fold
    fig, axes = plt.subplots(num_folds, 1, figsize=(12, 4 * num_folds))

    # Garantisce che `axes` sia sempre un array (anche con una sola fold)
    if num_folds == 1:
        axes = [axes]

    for i, history in enumerate(h_list):
        # Recupera la loss (MSE) e validation loss (val_MSE) dal dizionario `history`
        mse = history.history['mse']
        val_mse = history.history['val_mse']

        # Plot della Training Loss e Validation Loss
        axes[i].plot(mse, label='Training MSE', color='blue')
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

cup_csv = '../../data/ML-CUP24-TR.csv'

trainset = np.loadtxt(cup_csv, delimiter=",", skiprows=1, dtype=float)
trainset = trainset[:, 1:]

X = np.array(trainset[:, :-3])
y = np.array(trainset[:, -3:])

kfold = KFold(n_splits=4)

train_losses = []
val_losses = []
histories = []
n_fold = 1

for train_index, val_index in kfold.split(X):
    print(f"Training fold {n_fold}...")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = keras.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(3)
        ])

    opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.8, weight_decay=0.002)
    model.compile(loss='mse', metrics=['mse'], optimizer=opt)

    h = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0)
    train_scores = model.evaluate(X_train, y_train, verbose=0)
    val_scores = model.evaluate(X_val, y_val, verbose=0)

    train_losses.append(train_scores[0])
    val_losses.append(val_scores[0])
    histories.append(h)

    n_fold += 1

print('\n\nMedia e Deviazione Standard:')
print(f'Training MSE: {np.mean(train_losses):.4f} (±{np.std(train_losses):.4f})')
print(f'Validation MSE: {np.mean(val_losses):.4f} (±{np.std(val_losses):.4f})')

plot_histories(histories)