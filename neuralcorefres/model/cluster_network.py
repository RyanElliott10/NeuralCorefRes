import os
from typing import List

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Conv2D, CuDNNLSTM, Dense, Dropout, MaxPooling2D, Flatten, Reshape
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
        self._build_model()

    def load_saved(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def _build_model(self):
        self.model = Sequential()

        # CNN
        self.model.add(Conv2D(64, kernel_size=(8, 8), input_shape=(
            self.INPUT_MAXLEN, EMBEDDING_DIM, 2), activation='relu', strides=(1, 1)))
        self.model.add(MaxPooling2D(pool_size=(4, 4)))
        self.model.add(Flatten())
        self.model.add(Reshape(target_shape=(29, 6272)))

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

    @staticmethod
    def _pad_input_data(data: List[Tensor], maxlen: int) -> List[Tensor]:
        """
        Returns the data with each sentence containing 125 "words", where any
        sentences with fewer than 125 words are padded with a
        (2, EMBEDDING_DIM, POS_DIM) array.
        """
        for i, sentence in enumerate(data):
            for j in range(maxlen-sentence.shape[0]):
                data[i] = np.append(
                    data[i], [[np.zeros(maxlen), np.zeros(45)]], axis=0)

        return data

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
        # Fix padding for 125 word standard input
        self.xtrain = ClusterNetwork._pad_input_data(
            self.xtrain, self.INPUT_MAXLEN)
        self.xtest = ClusterNetwork._pad_input_data(
            self.xtest, self.INPUT_MAXLEN)

        assert self.xtrain[0].shape == (self.INPUT_MAXLEN, 2)
        assert self.xtrain[0][0].shape == (2,) # Number of attributes, can increase when more than just embedding and pos
        assert self.xtrain[0][0][0].shape == (EMBEDDING_DIM,)
        assert self.xtrain[0][0][1].shape == (45,)

        self.ytrain = sequence.pad_sequences(
            self.ytrain, maxlen=self.OUTPUT_LEN, dtype='float32', padding='post')
        self.ytest = sequence.pad_sequences(
            self.ytest, maxlen=self.OUTPUT_LEN, dtype='float32', padding='post')

        assert self.ytrain[0].shape == (self.OUTPUT_LEN,)

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
