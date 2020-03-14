from typing import List

from keras import Sequential
from keras.layers import (LSTM, Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
                          TimeDistributed)
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing import sequence

Tensor = List[float]
# XTrainData =


class CoreferenceNetwork:
    def __init__(self, xtrain, ytrain, xtest, ytest):
        pass

    def _build_model(self):
        pass

    def train(self):
        pass

    def predict(self, pred_data):
        pass
