import keras
import numpy as np
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from tabulate import tabulate


class MonkModel:
    def __init__(self, num_dataset=1):
        super().__init__()
        if num_dataset < 1 or num_dataset > 3:
            raise ValueError("num_dataset must be: 1, 2 or 3")

        trainset = np.loadtxt(f'../../data/monks-{num_dataset}.train', dtype=int, usecols=tuple(range(0, 7)))
        testset = np.loadtxt(f'../../data/monks-{num_dataset}.test', dtype=int, usecols=tuple(range(0, 7)))

        self.__X_input, self.__y_input = self.__create_in_out(trainset)
        self.__X_output, self.__y_output = self.__create_in_out(testset)
        self.__test_loss = None
        self.__test_accuracy = None
        self.__history = None
        self.__model = None
        self.__num_dataset = num_dataset

    def plot(self):
        if self.__history is None: raise ValueError("The model must be trained")

        history = self.__history

        plt.plot(history.history['loss'], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

    def train(self, epochs=300):
        X = self.__X_input
        y = self.__y_input

        model = keras.Sequential()
        model.add(keras.layers.Dense(8, input_dim=X.shape[1], activation='relu', kernel_regularizer = keras.regularizers.L2(0.03) if self.__num_dataset == 3 else None))
        model.add(keras.layers.Dense(8, activation='relu', kernel_regularizer = keras.regularizers.L2(0.02) if self.__num_dataset == 3 else None))
        model.add(keras.layers.Dense(1, activation='sigmoid', kernel_regularizer = keras.regularizers.L2(0.01) if self.__num_dataset == 3 else None))

        opt = keras.optimizers.SGD(learning_rate=0.015, momentum=0.9, weight_decay=0.001)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
        history = model.fit(X, y, epochs=epochs)

        self.__history = history

        self.__model = model

    def print_results(self):
        if self.__model is None: raise ValueError("The model must be trained before testing")
        model = self.__model

        self.__test_loss, self.__test_accuracy = model.evaluate(self.__X_output, self.__y_output)

        results = [["Loss", self.__test_loss],
                   ["Accuracy", self.__test_accuracy]]

        print("\n\n")
        print(tabulate(results, headers=["Metric", "Value"], tablefmt="github"))



    @staticmethod
    def __create_in_out(dataset):
        X = np.array([row[1:] for row in dataset])
        y = np.array([[row[0]] for row in dataset])

        X_one_hot = [to_categorical(X[:, i]) for i in range(X.shape[1])]
        X_one_hot = np.concatenate(X_one_hot, axis=1)
        return X_one_hot, y