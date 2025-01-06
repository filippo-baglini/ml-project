import keras
from keras_tuner import HyperModel, HyperParameters
from keras import layers

class MonkHyperModel(HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.__input_shape = input_shape

    def build(self, hp: HyperParameters):
        model = keras.Sequential([
            layers.Input(shape=self.__input_shape),
            layers.Dense(4, activation='tanh'),
            layers.Dense(1, activation='sigmoid')
        ])

        opt = keras.optimizers.SGD(
            learning_rate=hp.Float('learning_rate', min_value=0.01, max_value=0.09, step=0.01),
            momentum=hp.Float('momentum', min_value=0.1, max_value=0.9, step=0.1)
        )
        model.compile(loss='mse', metrics=[keras.metrics.BinaryAccuracy('accuracy')], optimizer=opt)
        return model