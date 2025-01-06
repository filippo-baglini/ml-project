import keras
from keras import layers
import numpy as np

from util.utils import Utils

cup_csv_train = '../../data/ML_Cup/ML-CUP24-TR.csv'
cup_csv_test = '../../data/ML_Cup/ML-CUP24-TS.csv'

trainset = np.loadtxt(cup_csv_train, delimiter=",", skiprows=1, dtype=float)
trainset = trainset[:, 1:]

X = np.array(trainset[:, :-3])
y = np.array(trainset[:, -3:])

model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(43, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(56, activation='tanh'),
    layers.Dense(3)
])

opt = keras.optimizers.Adam(learning_rate=0.004, weight_decay=0.0065, clipnorm=0.5)
model.compile(loss='mse', metrics=['mse', Utils.euclidean_distance], optimizer=opt)

h = model.fit(X, y, epochs=500)

print("Loss evaluate: " + str(model.evaluate(X, y)))
print("Loss history: " + str(h.history['loss'][-1]))
Utils.plot_history(h, metrics=['loss', 'euclidean_distance'])

testset = np.loadtxt(cup_csv_test, delimiter=",", skiprows=1, dtype=float)
X = testset[:, 1:]

y_pred = model.predict(X)
out = Utils.save_predictions(y_pred, folder="../..")

print(f"File salvato in: {out}")
