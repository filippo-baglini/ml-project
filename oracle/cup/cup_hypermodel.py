import keras
from keras import layers
import keras_tuner as kt
from keras_tuner import HyperParameters

class CupHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp: HyperParameters):
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=self.input_shape))

        # Prima layer Dense
        model.add(layers.Dense(
            units=201,
            activation='relu'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        # Seconda layer Dense
        model.add(layers.Dense(
            units=293,
            activation='relu'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        # Terza layer Dense
        model.add(layers.Dense(
            units=176,
            activation='relu'
        ))
        model.add(layers.BatchNormalization())

        # Output layer
        model.add(layers.Dense(3))

        optimizer = keras.optimizers.SGD(
            learning_rate=hp.Float('learning_rate', min_value=0.001, max_value=0.006, step=0.001),
            momentum=hp.Float('momentum', min_value=0.3, max_value=0.9, step=0.1),
            weight_decay=hp.Float('weight_decay', min_value=0.001, max_value=0.006, step=0.001),
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mse']
        )

        return model
