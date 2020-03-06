from typing import List

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

Tensor = List[float]


class Model():
    def __init__(self, input_maxlen: int, xtrain: List[Tensor], ytrain: List[Tensor], xtest: List[Tensor], ytest: List[Tensor]):
        self.input_maxlen = input_maxlen
        self.x_train = xtrain
        self.y_train = ytrain
        self.x_test = xtest
        self.y_test = ytest
        self._build_model()

    def _build_model(self):
        # Potential layer additions: ConvLSTM2D
        self.model = Sequential()
        self.model.add(Embedding(self.max_words, 512))
        self.model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.4))
        self.model.add(Dense(256, activation='softmax'))
        self.model.add(LSTM(256, dropout=0.1))
        self.model.add(Dense(128, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics='accuracy')

    def _pad_sequences(self):
        self.x_train = sequence.pad_sequences(
            self.x_train, maxlen=self.input_maxlen)
        self.x_test = sequence.pad_sequences(
            self.x_test, maxlen=self.input_maxlen)

    def train(self):
        self._pad_sequences()
        self.model.fit(self.x_train, self.y_train, epochs=5,
                       validation_data=(self.x_test, self.y_test))
