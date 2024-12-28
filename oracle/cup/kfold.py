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
histories = []
n_fold = 1

best_model: keras.Sequential = None

for train_index, val_index in kfold.split(X):
    print(f"Training fold {n_fold}...")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = keras.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(201, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(293, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(176, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(3)
        ])

    opt = keras.optimizers.SGD(learning_rate=0.004, momentum=0.7, weight_decay=0.005)
    model.compile(loss='mse', metrics=['mse'], optimizer=opt)

    early_stopping = EarlyStopping(monitor='val_loss', patience=19, restore_best_weights=True)

    h = model.fit(X_train, y_train, epochs=300, validation_data=(X_val, y_val), callbacks=[early_stopping])
    train_scores = model.evaluate(X_train, y_train, verbose=0)
    val_scores = model.evaluate(X_val, y_val, verbose=0)

    train_losses.append(train_scores[0])
    val_losses.append(val_scores[0])
    histories.append(h)

    n_fold += 1

print('\n\nMedia e Deviazione Standard:')
print(f'Training MSE: {np.mean(train_losses):.4f} (±{np.std(train_losses):.4f})')
print(f'Validation MSE: {np.mean(val_losses):.4f} (±{np.std(val_losses):.4f})')

Utils.plot_histories(histories)
