import os
from typing import List

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing import sequence

from neuralcorefres.model.word_embedding import EMBEDDING_DIM

# Enable AMD GPU usage
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

Tensor = List[float]


class ClusterNetwork():
    def __init__(self, xtrain: List[Tensor] = [], ytrain: List[Tensor] = [], xtest: List[Tensor] = [], ytest: List[Tensor] = [], inputmaxlen: int = 125, outputlen: int = 125):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.INPUT_MAXLEN = inputmaxlen
        self.OUTPUT_LEN = outputlen

        if len(self.xtrain) > 0:
            assert self.xtrain[0].shape == (self.INPUT_MAXLEN, 3, EMBEDDING_DIM)
            assert self.ytrain.shape == (self.xtrain.shape[0], self.OUTPUT_LEN)

        self._build_model()

    def load_saved(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def _build_model(self):
        INPUT_SHAPE = (self.INPUT_MAXLEN, 4, EMBEDDING_DIM)
        self.model = Sequential()

        # CNN
        self.model.add(Conv2D(32, kernel_size=(3, 5), padding='same', activation='relu', input_shape=INPUT_SHAPE))
        self.model.add(MaxPooling2D(2))
        self.model.add(TimeDistributed(Flatten()))

        # RNN
        self.model.add(LSTM(1028, return_sequences=True, dropout=0.5, activation='relu'))
        self.model.add(LSTM(512, dropout=0.3, activation='tanh'))

        # Dense
        self.model.add(Dense(512, activation='tanh'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.OUTPUT_LEN, activation='tanh'))

        opt = RMSprop(learning_rate=1e-3)
        self.model.compile(loss='logcosh', optimizer=opt, metrics=['acc'])
        self.model.summary()

    def _pad_sequences(self):
        self.ytrain = sequence.pad_sequences(self.ytrain, maxlen=self.OUTPUT_LEN, dtype='float32', padding='post')
        self.ytest = sequence.pad_sequences(self.ytest, maxlen=self.OUTPUT_LEN, dtype='float32', padding='post')

        assert self.xtrain[0].shape == (self.INPUT_MAXLEN, 3, EMBEDDING_DIM)
        assert self.ytrain.shape == (self.xtrain.shape[0], self.OUTPUT_LEN)

    def train(self):
        self._pad_sequences()
        self.model.fit(self.xtrain, self.ytrain, epochs=3)
        self.model.save(".././data/models/clusters/small.h5")
        score, acc = self.model.evaluate(self.xtest, self.ytest)
        print(score, acc)

    def predict(self, embeddings: Tensor, padded_pos: Tensor):
        padded_embeddings = np.asarray(sequence.pad_sequences(
            [embeddings], maxlen=self.INPUT_MAXLEN, dtype='float32'))
        print(padded_embeddings.shape)
        print(padded_pos.shape)
        # return self.model.predict(padded_embeddings)
