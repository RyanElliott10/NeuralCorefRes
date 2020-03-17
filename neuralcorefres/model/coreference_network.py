# -*- coding: utf-8 -*-
# CNN -> LSTM -> Dense network for identifying similar mention clusters
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import os
import pprint
import random
import sys
from collections import defaultdict, namedtuple
from itertools import chain
from multiprocessing import Pool
from typing import DefaultDict, List, Tuple

import numpy as np
import spacy
import tensorflow as tf
from progress.bar import IncrementalBar
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (LSTM, BatchNormalization, Conv2D, Dense,
                                     Flatten, MaxPooling2D, TimeDistributed)

from neuralcorefres.model.word_embedding import EMBEDDING_DIM, WordEmbedding
from neuralcorefres.parsedata.parse_clusters import ParseClusters
from neuralcorefres.parsedata.preco_parser import PreCoParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(threshold=sys.maxsize)
pretty_printer = pprint.PrettyPrinter()

Tensor = List[float]
SentenceAttributes = namedtuple('SentenceAttributes', ['we', 'pos_onehot', 'dep_embeddings', 'deps_onehot'])
EntityAtttibutes = namedtuple('EntityAtttibutes', ['entity_we', 'headword_embed', 'headword_relation'])
TrainingSampleAttributes = namedtuple('TrainingSampleAttributes', ['we', 'pos_onehot', 'dep_embeddings', 'deps_onehot', 'entity_we', 'headword_embed',
                                                                   'headword_relation'])


class CoreferenceNetwork:
    def __init__(self, x_train: List[Tensor] = [], y_train: List[Tensor] = [], x_test: List[Tensor] = [], y_test: List[Tensor] = [], inputmaxlen: int = 128, outputlen: int = 128):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.INPUT_MAXLEN = inputmaxlen
        self.OUTPUT_LEN = outputlen
        self.model = None

        if len(self.x_train) > 0:
            self._build_model()

            assert self.x_train[0].shape == (7, self.INPUT_MAXLEN, EMBEDDING_DIM)
            assert self.y_train.shape == (self.x_train.shape[0], self.OUTPUT_LEN)

    def load_saved(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def _build_model(self):
        INPUT_SHAPE = (7, self.INPUT_MAXLEN, EMBEDDING_DIM)
        CHAN_DIM = -1
        self.model = tf.keras.models.Sequential()

        # CNN
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization(axis=CHAN_DIM))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(TimeDistributed(Flatten()))

        # RNN
        self.model.add(LSTM(256, return_sequences=True))
        self.model.add(LSTM(64))

        # Dense
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.OUTPUT_LEN, activation='sigmoid'))

        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(loss=tf.keras.metrics.binary_crossentropy, optimizer=opt,
                           metrics=[tf.keras.metrics.categorical_accuracy, 'acc'])
        self.model.summary()

    def train(self, generate_data=None):
        if generate_data is not None:
            self.model.fit(generate_data(), steps_per_epoch=1000, epochs=38, verbose=1)
        else:
            self.model.fit(self.x_train, self.y_train, epochs=10)

        self.model.save('../data/models/clusters/small.h5')
        return self.model.evaluate(self.x_test, self.y_test)

    def get_model(self):
        if not self.model:
            self.load_saved('../data/models/clusters/small.h5')

    def predict(self, sent: str):
        entities = ParseClusters.get_named_entities(sent)

        embedding_model = WordEmbedding(model_path='../data/models/word_embeddings/google-vectors.model')

        tokenized = ParseClusters.tokenize_sent(sent)
        pos_onehot_map = PreCoParser.get_pos_onehot_map()

        gen_sent_features = CoreferenceNetwork._sent_to_we_pos_em([tokenized], embedding_model, pos_onehot_map)
        embed_dict = CoreferenceNetwork._sent_to_we_pos_em([tokenized], embedding_model, pos_onehot_map)

        print(entities)

        queue = []
        for entity in entities.items():
            queue.append(CoreferenceNetwork._sent_to_input(sent, entity, embedding_model, embed_dict))

        inputs = np.asarray([np.asarray((*gen_sent_features[0], *specific_features)) for specific_features in queue])

        self.get_model()
        predictions = self.model.predict(inputs)
        return [(list(entities.keys())[i], predictions[i][:len(tokenized)]) for i in range(len(predictions))]

    # TODO convert these to util.py

    @staticmethod
    def _sent_to_input(sent: str, entity: Tuple[str, List[int]], embedding_model, embed_dict):
        # VERY naive appraoch, only gets the relation for the first occurence of a particular word
        indices = (0, *entity[1][0])
        raw_entity = entity[0]
        return CoreferenceNetwork._get_entity_specific_feature(raw_entity, embedding_model, indices, embed_dict)

    @staticmethod
    def _pad_1d_tensor(pos_onehot, maxlen: int, dtype: str = 'float16', value: float = 0.0):
        return tf.keras.preprocessing.sequence.pad_sequences(pos_onehot, maxlen, dtype=dtype, padding='post', value=value)

    @staticmethod
    def _get_nn_friendly_output(data):
        [[d.pop(0) for d in dp if len(d) >= 2] for dp in data]
        data = [list(chain.from_iterable(el)) for el in data]
        data = CoreferenceNetwork._pad_1d_tensor(data, 128, dtype='int32', value=-1.0)
        return [tf.reduce_max(hots, axis=0) for hots in tf.one_hot(data, 128)]

    @staticmethod
    def _sent_to_we_pos_em(sents: List[List[str]], embedding_model, pos_onehot_map) -> DefaultDict[int, SentenceAttributes]:
        """ Returns a dictionary with key as index (maybe not safe? perhaps use ' '.join(sent)) containing word embeddings.  """

        # TODO remove dict dependency since keys are indexes, anyway
        ret_dict = defaultdict(list)
        for i, sent in enumerate(sents):
            features = np.zeros((4, 128, EMBEDDING_DIM))

            we = PreCoParser.get_embedding_for_sent(sent, embedding_model)
            pos_onehot = CoreferenceNetwork._pad_1d_tensor(
                PreCoParser.get_pos_onehot_map_for_sent(sent, pos_onehot_map), we.shape[1])
            dep_embeddings = np.asarray([PreCoParser.get_dep_embeddings(sent, embedding_model)])  # Horribly slow
            deps_onehot = CoreferenceNetwork._pad_1d_tensor(PreCoParser.get_deps_onehot(sent), we.shape[1])  # Horribly slow

            features[0][:we.shape[0], :we.shape[1]] = we
            features[1][:pos_onehot.shape[0], :pos_onehot.shape[1]] = pos_onehot
            features[2][:dep_embeddings.shape[1], :dep_embeddings.shape[2]] = dep_embeddings
            features[3][:deps_onehot.shape[0], :deps_onehot.shape[1]] = deps_onehot

            assert features[0].shape == (128, EMBEDDING_DIM)
            assert features[1].shape == (128, EMBEDDING_DIM)
            assert features[2].shape == (128, EMBEDDING_DIM)
            assert features[3].shape == (128, EMBEDDING_DIM)

            ret_dict[i] = SentenceAttributes(*features)

        return ret_dict

    @staticmethod
    def _get_entity_specific_feature(entity, embedding_model, indices, embed_dict):
        features = np.zeros((3, 128, EMBEDDING_DIM))

        we = PreCoParser.get_embedding_for_sent(entity, embedding_model)
        dep_embeddings = embed_dict[indices[0]].dep_embeddings[indices[1]:indices[2]]
        dep_rel = embed_dict[indices[0]].deps_onehot[indices[1]:indices[2]]

        features[0][:we.shape[0], :we.shape[1]] = we
        features[1][:dep_embeddings.shape[0], :dep_embeddings.shape[1]] = dep_embeddings
        features[2][:dep_rel.shape[0], :dep_rel.shape[1]] = dep_rel

        assert features[0].shape == (128, EMBEDDING_DIM)
        assert features[1].shape == (128, EMBEDDING_DIM)
        assert features[2].shape == (128, EMBEDDING_DIM)

        return EntityAtttibutes(*features)

    @staticmethod
    def _single_custom_cluster_to_nn(sents, clusters, embed_dict, embedding_model):
        """
        Iterate through clusters and get the word embeddings for each cluster as shape:
            (clusters_len, num_in_cluster, EMBEDDONG_DIM)
            [ [ [], [], [], [] ] ]
        """
        features = []
        for key, cluster in clusters.items():
            for indices in cluster:
                sub = sents[indices[0]][indices[1]:indices[2]]
                sent_clusters = list(filter(lambda x: x[0] == indices[0], cluster))
                features.append((indices[0], sent_clusters, CoreferenceNetwork._get_entity_specific_feature(
                    sub, embedding_model, indices, embed_dict)))

        return features

    @staticmethod
    def expand_cluster(cluster: List[List[int]]) -> List[List[int]]:
        """
        Modifies each cluster by converting the second and third elements into the range between them.
        [0, 2, 5] -> [0, 2, 3, 4]
        """
        return [[c[0], *list(range(c[1], c[2]))].copy() for c in cluster]

    @staticmethod
    def custom_cluster_to_nn_input(sents, clusters):
        """
        Preprocessing steps:
            * Grab entites from sentence using ParseClusters.get_named_entities(sent)
            * Features for NN: Shape: (6, 128, 400, 128) -> (num_features, max_sent_len, EMBEDDING_DIM, max_sent_len)
                * Raw sentence word embeddings
                * POS of every word in sentence
                * Headword relations for every word in sentence
                ***************** WORD SPECIFIC *****************
                * Word embedding(s) for current entity
                * Headword word embedding for the current entity
                * Headword relation for the current entity
        """
        assert len(sents) == len(clusters)

        embedding_model = WordEmbedding(model_path='.././data/models/word_embeddings/google-vectors.model')
        pos_onehot_map = PreCoParser.get_pos_onehot_map()

        bar = IncrementalBar(' *\tGenerating NN friendly input from custom clusters and sentences', max=len(sents))
        xy_split = []
        x_train = []
        y_train = []

        for i in range(len(sents)):
            embed_dict = CoreferenceNetwork._sent_to_we_pos_em(sents[i], embedding_model, pos_onehot_map)
            indiv_features = CoreferenceNetwork._single_custom_cluster_to_nn(
                sents[i], clusters[i], embed_dict, embedding_model)
            sent_vals = list(embed_dict.values())

            xy_split.extend([(TrainingSampleAttributes(*sent_vals[idx], indiv.entity_we, indiv.headword_embed,
                                                       indiv.headword_relation), CoreferenceNetwork.expand_cluster(cluster)) for idx, cluster, indiv in indiv_features])
            bar.next()

        print()
        random.seed(42)
        random.shuffle(xy_split)
        x_train, y_train = zip(*xy_split)
        y_train = CoreferenceNetwork._get_nn_friendly_output(y_train)

        print(f' *\tTotal training samples: {len(x_train)}')
        return np.asarray(x_train), np.asarray(y_train)
