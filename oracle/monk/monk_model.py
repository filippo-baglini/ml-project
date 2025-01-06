import keras
from keras import layers
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate

from keras import callbacks
from keras import backend as be

from util.utils import Utils


class MonkModel:
    def __init__(self, num_dataset=1):
        super().__init__()
        if num_dataset < 1 or num_dataset > 3:
            raise ValueError("num_dataset must be: 1, 2 or 3")

        trainset = np.loadtxt(f'../../data/monks-{num_dataset}.train', dtype=int, usecols=tuple(range(0, 7)))
        testset = np.loadtxt(f'../../data/monks-{num_dataset}.test', dtype=int, usecols=tuple(range(0, 7)))

        self.X_train, self.y_train = self.__create_in_out(trainset)
        self.__X_test, self.__y_test = self.__create_in_out(testset)
        self.__test_loss = None
        self.__test_accuracy = None
        self.__history = None
        self.__model = None
        self.__num_dataset = num_dataset

    def train(self, epochs=300, kfold=False, splits=4):
        if kfold:
            self.__kfold(epochs=epochs, splits=splits)
            return

        X = self.X_train
        y = self.y_train

        model = self.__build_model()
        history = model.fit(X, y, epochs=epochs)

        self.__history = history

        self.__model = model

    def __kfold(self, epochs=300, splits=4):
        kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        X = self.X_train
        y = self.y_train

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        histories = []

        n_fold = 1

        for train_index, val_index in kfold.split(X, y):
            print(f"Training fold {n_fold}...")

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            print(f"Train class distribution: {np.bincount(y_train.ravel().astype(int))}")
            print(f"Validation class distribution: {np.bincount(y_val.ravel().astype(int))}\n\n")

            model = self.__build_model()

            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, min_lr=0.001)

            h = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[reduce_lr, early_stopping], verbose=0)

            train_scores = model.evaluate(X_train, y_train, verbose=0)
            val_scores = model.evaluate(X_val, y_val, verbose=0)

            print(f"Train scores: {train_scores}")
            print(f"Validation scores: {val_scores}")

            train_losses.append(train_scores[0])
            val_losses.append(val_scores[0])

            train_accuracies.append(train_scores[1])
            val_accuracies.append(val_scores[1])
            histories.append(h)

            be.clear_session()
            n_fold += 1

        print('\n\nLoss (mean):')
        print(f'Training Loss: {np.mean(train_losses):.4f}')
        print(f'Validation Loss: {np.mean(val_losses):.4f}')

        print('\n\nAccuracies (mean):')
        print(f'Training Accuracy: {np.mean(train_accuracies):.4f}')
        print(f'Validation Accuracy: {np.mean(val_accuracies):.4f}')

        Utils.plot_histories(histories, metrics=['loss', 'accuracy'])

    def print_results(self, plot=True):
        if self.__model is None: raise ValueError("The model must be trained before testing")
        model = self.__model

        self.__test_loss, self.__test_accuracy = model.evaluate(self.__X_test, self.__y_test)

        results = [["Loss", self.__test_loss],
                   ["Accuracy", self.__test_accuracy]]

        print("\n\n")
        print(tabulate(results, headers=["Metric", "Value"], tablefmt="github"))

        if plot:
            Utils.plot_history(self.__history, metrics=['loss', 'accuracy'])

    @staticmethod
    def __create_in_out(dataset):
        X = dataset[:, 1:]
        y = dataset[:, :1]

        encoder = OneHotEncoder(sparse_output=False)
        X_one_hot = encoder.fit_transform(X)

        # X_one_hot = [to_categorical(X[:, i]) for i in range(X.shape[1])]
        # X_one_hot = np.concatenate(X_one_hot, axis=1)
        return X_one_hot, y

    def __build_model(self):
        X = self.X_train

        model = keras.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(6, activation='relu', kernel_regularizer=keras.regularizers.L2(0.002) if self.__num_dataset == 3 else None),
            layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(0.001) if self.__num_dataset == 3 else None)
        ])

        opt = keras.optimizers.SGD(learning_rate=0.0525, momentum=0.825, weight_decay=0.002)
        model.compile(loss='mse', metrics=[keras.metrics.BinaryAccuracy('accuracy')], optimizer=opt)
        return model