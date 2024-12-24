import keras
from keras import layers
import keras_tuner as kt


class CupHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=self.input_shape))

        # Prima layer Dense
        model.add(layers.Dense(
            units=hp.Choice('units_1', values=[2, 10, 20, 32, 64, 128, 256]),
            activation=hp.Choice('activation_1', values=['relu', 'sigmoid', 'softmax', 'tanh'])
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

        # Seconda layer Dense
        model.add(layers.Dense(
            units=hp.Choice('units_2', values=[2, 10, 20, 32, 64, 128, 256]),
            activation=hp.Choice('activation_2', values=['relu', 'sigmoid', 'softmax', 'tanh'])
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.4, step=0.1)))

        # Terza layer Dense
        model.add(layers.Dense(
            units=hp.Choice('units_3', values=[2, 10, 20, 32, 64, 128, 256]),
            activation=hp.Choice('activation_3', values=['relu', 'sigmoid', 'softmax', 'tanh'])
        ))
        model.add(layers.BatchNormalization())

        # Output layer
        model.add(layers.Dense(3))

        # Configurazione dell'ottimizzatore
        # Learning rate: da 0.0001 a 0.01 con incrementi in scala logaritmica
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value=0.005,
                                    max_value=0.015,
                                    step=0.001)

        # Momentum: da 0.6 a 0.9 con step di 0.1
        hp_momentum = hp.Float('momentum',
                               min_value=0.4,
                               max_value=0.9,
                               step=0.1)

        # Weight decay: da 0.001 a 0.005 con step di 0.001
        hp_weight_decay = hp.Float('weight_decay',
                                   min_value=0.001,
                                   max_value=0.005,
                                   step=0.001)

        optimizer = keras.optimizers.SGD(
            learning_rate=hp_learning_rate,
            momentum=hp_momentum,
            weight_decay=hp_weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mse']
        )

        return model
