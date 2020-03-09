import os
from typing import List

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Conv1D, CuDNNLSTM, Dense, Dropout, MaxPooling1D
from keras.optimizers import Adam
from keras.preprocessing import sequence

from neuralcorefres.model.word_embedding import EMBEDDING_DIM

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

Tensor = List[float]


class ClusterNetwork():
    def __init__(self, xtrain: List[Tensor] = [], ytrain: List[Tensor] = [], xtest: List[Tensor] = [], ytest: List[Tensor] = [], inputmaxlen: int = 125, outputlen: int = 128):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.INPUT_MAXLEN = inputmaxlen
        self.OUTPUT_LEN = outputlen
        self._build_model()

    def load_saved(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def _build_model(self):
        self.model = Sequential()

        # CNN
        self.model.add(Conv1D(64, kernel_size=(8), input_shape=(
            self.INPUT_MAXLEN, EMBEDDING_DIM), padding='causal', activation='relu', strides=1))
        self.model.add(MaxPooling1D(pool_size=(4)))

        # LSTM
        # self.model.add(LSTM(512, input_shape=(self.INPUT_MAXLEN,
        #                                       EMBEDDING_DIM), dropout=0.2, activation='tanh'))
        self.model.add(LSTM(512, return_sequences=False,
                            dropout=0.2, activation='tanh'))
        # self.model.add(LSTM(256, dropout=0.2, activation='tanh'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Dense(self.OUTPUT_LEN, activation='tanh'))

        opt = Adam(lr=1e-3, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt, metrics=['accuracy'])
        print(self.model.summary())

    def _pad_sequences(self):
        # self.xtrain = [sequence.pad_sequences(data, maxlen=self.]
        # self.xtrain = sequence.pad_sequences(
        #     self.xtrain, maxlen=self.INPUT_MAXLEN, dtype='float32', padding='post')
        self.xtest = sequence.pad_sequences(
            self.xtest, maxlen=self.INPUT_MAXLEN, dtype='float32', padding='post')
        self.ytrain = sequence.pad_sequences(
            self.ytrain, maxlen=self.OUTPUT_LEN, dtype='float32', padding='post')
        self.ytest = sequence.pad_sequences(
            self.ytest, maxlen=self.OUTPUT_LEN, dtype='float32', padding='post')

        # print(self.ytrain[0])
        # print(self.xtrain[0])
        # for el in self.xtrain[0]:
        #     print(el)

    def train(self):
        self._pad_sequences()
        self.model.fit(self.xtrain, self.ytrain, epochs=1)
        self.model.save(".././data/models/clusters/small.h5")
        score, acc = self.model.evaluate(self.xtest, self.ytest)
        print(score, acc)

    def predict(self, embeddings: Tensor):
        padded = np.asarray(sequence.pad_sequences(
            [embeddings], maxlen=self.INPUT_MAXLEN))
        padded = padded.astype(np.float)
        print(padded.shape)
        return self.model.predict(padded)
