import os
from typing import List

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import (LSTM, Conv2D, CuDNNLSTM, Dense, Dropout, Flatten, Conv3D, MaxPooling3D,
                          MaxPooling2D, TimeDistributed)
from keras.optimizers import Adam
from keras.preprocessing import sequence

from neuralcorefres.model.word_embedding import EMBEDDING_DIM

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

        assert self.xtrain[0].shape == (self.INPUT_MAXLEN, 2, EMBEDDING_DIM)
        assert self.ytrain.shape == (self.xtrain.shape[0], self.OUTPUT_LEN)

        self._build_model()

    def load_saved(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def _build_model(self):
        self.model = Sequential()

        # CNN
        self.model.add(Conv2D(32, kernel_size=5, strides=(1), padding='same', input_shape=(self.INPUT_MAXLEN, 2, EMBEDDING_DIM)))
        # self.model.add(Conv3D(16, kernel_size=(3, 3, 3)))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(TimeDistributed(Flatten()))

        # LSTM
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
        """
        (n_samples, n_words, n_attributes (word embedding, pos, etc))
        [ [ word_embedding, pos ] ]

        xtrain[sentence_sample][word_position][attribute]
        xtrain[0][0] -> first word's attributes in first sentence
        xtrain[37][5] -> sixth word's attributes in 38th sentence
        xtrain[0][0][0] -> word_embedding
        xtrain[0][0][1] -> pos one-hot encoding
        """
        self.ytrain = sequence.pad_sequences(
            self.ytrain, maxlen=self.OUTPUT_LEN, dtype='float32', padding='post')
        self.ytest = sequence.pad_sequences(
            self.ytest, maxlen=self.OUTPUT_LEN, dtype='float32', padding='post')

        assert self.xtrain[0].shape == (self.INPUT_MAXLEN, 2, EMBEDDING_DIM)

        # print("\nXTRAIN FINAL SHAPE:", self.xtrain.shape)
        # print("\nYTRAIN FINAL SHAPE:", self.ytrain.shape)

        # Current issue: this fails because some data isn't being parsed and prepared correctly
        # assert self.ytrain.shape == (self.xtrain.shape[0], self.OUTPUT_LEN)

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
