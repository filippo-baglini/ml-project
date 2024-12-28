import keras
from keras import layers
import numpy as np

from util.utils import Utils

cup_csv_train = '../../data/ML-CUP24-TR.csv'
cup_csv_test = '../../data/ML-CUP24-TS.csv'

trainset = np.loadtxt(cup_csv_train, delimiter=",", skiprows=1, dtype=float)
trainset = trainset[:, 1:]

X = np.array(trainset[:, :-3])
y = np.array(trainset[:, -3:])

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

opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.7, weight_decay=0.005)
model.compile(loss='mse', metrics=['mse', Utils.euclidean_distance], optimizer=opt)

h = model.fit(X, y, epochs=300)

print("Loss: " + str(model.evaluate(X, y)))
Utils.plot_history(h)

testset = np.loadtxt(cup_csv_test, delimiter=",", skiprows=1, dtype=float)
X = testset[:, 1:]

y_pred = model.predict(X)
out = Utils.save_predictions(y_pred, folder="../..")

print(f"File salvato in: {out}")