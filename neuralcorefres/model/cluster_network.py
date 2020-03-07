from typing import List

import tensorflow as tf
from keras.preprocessing import sequence
from tensorflow.compat.v1.keras.layers import (LSTM, CuDNNLSTM, Dense, Dropout,
                                               Embedding)
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from neuralcorefres.model.word_embedding import EMBEDDING_DIM

Tensor = List[float]


class ClusterNetwork():
    def __init__(self, xtrain: List[Tensor], ytrain: List[Tensor], xtest: List[Tensor], ytest: List[Tensor], inputmaxlen: int = 125, outputlen: int = 128):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.INPUT_MAXLEN = inputmaxlen
        self.OUTPUT_LEN = outputlen
        self._build_model()

    def _build_model(self):
        # Potential layer additions: ConvLSTM2D
        self.model = Sequential()
        self.model.add(LSTM(512, input_shape=(self.INPUT_MAXLEN, EMBEDDING_DIM)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.OUTPUT_LEN, activation='softmax'))
        self.model.add(Dropout(0.2))

        opt = Adam(lr=1e-3, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt, metrics=['accuracy'])
        print(self.model.summary())

    def _pad_sequences(self):
        self.xtrain = sequence.pad_sequences(
            self.xtrain, maxlen=self.INPUT_MAXLEN)
        self.xtest = sequence.pad_sequences(
            self.xtest, maxlen=self.INPUT_MAXLEN)
        self.ytrain = sequence.pad_sequences(
            self.ytrain, maxlen=self.OUTPUT_LEN)
        self.ytest = sequence.pad_sequences(self.ytest, maxlen=self.OUTPUT_LEN)

    def train(self):
        self._pad_sequences()
        self.model.fit(self.xtrain, self.ytrain, epochs=5)
