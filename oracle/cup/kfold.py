import numpy as np
import keras
from keras import layers
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import KFold

from util.utils import Utils

cup_csv = '../../data/ML-CUP24-TR.csv'

trainset = np.loadtxt(cup_csv, delimiter=",", skiprows=1, dtype=float)
trainset = trainset[:, 1:]

X = np.array(trainset[:, :-3])
y = np.array(trainset[:, -3:])

kfold = KFold(n_splits=4)

train_losses = []
val_losses = []

train_losses_ed = []
val_losses_ed = []

histories = []
n_fold = 1

for train_index, val_index in kfold.split(X):
    print(f"Training fold {n_fold}...")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(43, activation='tanh'),
        layers.Dropout(0.2),
        layers.Dense(56, activation='tanh'),
        layers.Dense(3)
    ])

    opt = keras.optimizers.Adam(learning_rate=0.004, weight_decay=0.0065, clipnorm=0.5)
    model.compile(loss='mse', metrics=['mse',Utils.euclidean_distance], optimizer=opt)

    early_stopping = EarlyStopping(monitor='val_loss', patience=19, restore_best_weights=True)

    h = model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
    train_scores = model.evaluate(X_train, y_train, verbose=0)
    val_scores = model.evaluate(X_val, y_val, verbose=0)

    train_losses.append(train_scores[0])
    val_losses.append(val_scores[0])

    train_losses_ed.append(train_scores[2])
    val_losses_ed.append(val_scores[2])

    histories.append(h)

    n_fold += 1
    print(f"Train scores: {train_scores}")
    print(f"Val scores: {val_scores}\n")

print('\n\nMedia:')
print(f'Training MSE: {np.mean(train_losses):.4f}')
print(f'Validation MSE: {np.mean(val_losses):.4f}\n')
print(f'Training Euclidean Distance: {np.mean(train_losses_ed):.4f}')
print(f'Validation Euclidean Distance: {np.mean(val_losses_ed):.4f}\n')


Utils.plot_histories(histories, metrics=['mse', 'euclidean_distance'])