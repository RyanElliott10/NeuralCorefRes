import os
from typing import List

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import (LSTM, Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
                          TimeDistributed)
from keras.optimizers import RMSprop, SGD
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

        if len(self.xtrain) > 0:
            assert self.xtrain[0].shape == (self.INPUT_MAXLEN, 2, EMBEDDING_DIM)
        if len(self.ytrain) > 0:
            assert self.ytrain.shape == (self.xtrain.shape[0], self.OUTPUT_LEN)

        self._build_model()

    def load_saved(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def _build_model(self):
        self.model = Sequential()

        # CNN
        self.model.add(Conv2D(filters=32, kernel_size=(3, 5), padding='same',
                              activation='tanh', input_shape=(self.INPUT_MAXLEN, 2, EMBEDDING_DIM)))
        self.model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='tanh'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(TimeDistributed(Flatten()))

        # LSTM
        self.model.add(LSTM(512, dropout=0.2, return_sequences=False))
        # self.model.add(LSTM(256, dropout=0.2, return_sequences=True))
        # self.model.add(LSTM(128, dropout=0.2))
        self.model.add(Dense(512, activation='tanh'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.OUTPUT_LEN, activation='tanh'))

        # opt = RMSprop(learning_rate=1e-3)
        opt = RMSprop()
        # opt = SGD()
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

    def predict(self, embeddings: Tensor, padded_pos: Tensor):
        padded_embeddings = np.asarray(sequence.pad_sequences(
            [embeddings], maxlen=self.INPUT_MAXLEN, dtype='float32'))
        print(padded_embeddings.shape)
        print(padded_pos.shape)
        # return self.model.predict(padded_embeddings)
