import keras
from keras import layers
import keras_tuner as kt
from keras_tuner import HyperParameters

class CupHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp: HyperParameters):
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Dense(
                43,
                activation='tanh',
            ),
            layers.Dropout(0.2),
            layers.Dense(
                56,
                activation='tanh',
            ),
            layers.Dense(3)
        ])

        optimizer = keras.optimizers.SGD(
            learning_rate=0.003,
            momentum=0.9,
            weight_decay=hp.Float('weight_decay', min_value=0.004, max_value=0.008, step=0.001),
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mse']
        )

        return model
