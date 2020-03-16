import pprint
import sys
from collections import defaultdict, namedtuple
from itertools import chain
from typing import DefaultDict, List, Tuple

import numpy as np
import spacy
import tensorflow as tf
from keras import Sequential
from keras.layers import (LSTM, Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
                          TimeDistributed)
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing import sequence
from progress.bar import IncrementalBar

from neuralcorefres.model.word_embedding import EMBEDDING_DIM, WordEmbedding
from neuralcorefres.parsedata.parse_clusters import ParseClusters
from neuralcorefres.parsedata.preco_parser import PreCoParser

np.set_printoptions(threshold=sys.maxsize)
pretty_printer = pprint.PrettyPrinter()

Tensor = List[float]
SentenceAttributes = namedtuple('SentenceAttributes', ['we', 'pos_onehot', 'dep_embeddings', 'deps_onehot'])
EntityAtttibutes = namedtuple('EntityAtttibutes', ['entity_we', 'headword_embed', 'headword_relation'])
TrainingSampleAttributes = namedtuple('TrainingSampleAttributes', ['we', 'pos_onehot', 'dep_embeddings', 'deps_onehot', 'entity_we', 'headword_embed',
                                                                   'headword_relation'])


class CoreferenceNetwork:
    def __init__(self, xtrain: List[Tensor] = [], ytrain: List[Tensor] = [], xtest: List[Tensor] = [], ytest: List[Tensor] = [], inputmaxlen: int = 125, outputlen: int = 125):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.INPUT_MAXLEN = inputmaxlen
        self.OUTPUT_LEN = outputlen
        self.model = None

        if len(self.xtrain) > 0:
            self._build_model()

            assert self.xtrain[0].shape == (7, self.INPUT_MAXLEN, EMBEDDING_DIM)
            assert self.ytrain.shape == (self.xtrain.shape[0], self.OUTPUT_LEN)

    def load_saved(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def _build_model(self):
        INPUT_SHAPE = (7, self.INPUT_MAXLEN, EMBEDDING_DIM)
        self.model = Sequential()

        # CNN
        self.model.add(Conv2D(32, kernel_size=(3, 5), padding='same', activation='tanh', input_shape=INPUT_SHAPE))
        self.model.add(MaxPooling2D(2))
        self.model.add(TimeDistributed(Flatten()))

        # RNN
        self.model.add(LSTM(1028, return_sequences=True, dropout=0.5, activation='tanh'))
        self.model.add(LSTM(512, dropout=0.3, activation='tanh'))

        # Dense
        self.model.add(Dense(512, activation='tanh'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.OUTPUT_LEN, activation='tanh'))

        opt = RMSprop(learning_rate=1e-3)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
        self.model.summary()

    def train(self):
        self.model.fit(self.xtrain, self.ytrain, epochs=3)
        self.model.save('.././data/models/clusters/small.h5')
        # score, acc = self.model.evaluate(self.xtest, self.ytest)
        # print(score, acc)

    def get_model(self):
        if not self.model:
            self.load_saved('.././data/models/clusters/small.h5')

    def predict(self, sent: str):
        entities = ParseClusters.get_named_entities(sent)

        embedding_model = WordEmbedding(model_path='.././data/models/word_embeddings/google-vectors.model')

        tokenized = ParseClusters.tokenize_sent(sent)
        pos_onehot_map = PreCoParser.get_pos_onehot_map()

        gen_sent_features = CoreferenceNetwork._sent_to_we_pos_em(
            [tokenized], embedding_model, pos_onehot_map)
        embed_dict = CoreferenceNetwork._sent_to_we_pos_em([tokenized], embedding_model, pos_onehot_map)

        print(entities)

        queue = []
        for entity in entities.items():
            queue.append(CoreferenceNetwork._sent_to_input(sent, entity, embedding_model, embed_dict))

        inputs = np.asarray([np.asarray((*gen_sent_features[0], *specific_features)) for specific_features in queue])

        print(inputs.shape)
        print(inputs[0].shape)

        self.get_model()
        predictions = self.model.predict(inputs)
        print(predictions.shape)
        print(predictions[3])

    # TODO convert these to util.py

    @staticmethod
    def _sent_to_input(sent: str, entity: Tuple[str, List[int]], embedding_model, embed_dict):
        print('entity:', entity)
        # VERY naive appraoch, only gets the relation for the first occurence of a particular word
        indices = (0, *entity[1][0])
        print('indices:', indices)
        raw_entity = entity[0]
        return CoreferenceNetwork._get_entity_specific_feature(raw_entity, embedding_model, indices, embed_dict)

    @staticmethod
    def _pad_1d_tensor(pos_onehot, maxlen: int, dtype: str = 'float32', value: float = 0.0):
        return sequence.pad_sequences(pos_onehot, maxlen, dtype=dtype, padding='post', value=value)

    @staticmethod
    def _get_nn_friendly_output(data):
        [[d.pop(0) for d in dp if len(d) == 3] for dp in data]
        data = [list(chain.from_iterable(el)) for el in data]
        data = CoreferenceNetwork._pad_1d_tensor(data, 125, dtype='int32', value=-1.0)
        return [tf.reduce_max(hots, axis=0) for hots in tf.one_hot(data, 125)]

    @staticmethod
    def _sent_to_we_pos_em(sents: List[List[str]], embedding_model, pos_onehot_map) -> DefaultDict[int, SentenceAttributes]:
        """ Returns a dictionary with key as index (maybe not safe? perhaps use ' '.join(sent)) containing word embeddings.  """

        # TODO remove list dependency since keys are indexes, anyway
        ret_dict = defaultdict(list)
        for i, sent in enumerate(sents):
            features = np.zeros((4, 125, EMBEDDING_DIM))

            we = np.asarray(PreCoParser.get_embedding_for_sent(sent, embedding_model))
            pos_onehot = np.asarray(CoreferenceNetwork._pad_1d_tensor(
                PreCoParser.get_pos_onehot_map_for_sent(sent, pos_onehot_map), we.shape[1]))
            dep_embeddings = np.asarray([PreCoParser.get_dep_embeddings(sent, embedding_model)])  # Horribly slow
            deps_onehot = np.asarray(CoreferenceNetwork._pad_1d_tensor(
                PreCoParser.get_deps_onehot(sent), we.shape[1]))  # Horribly slow

            features[0][:we.shape[0], :we.shape[1]] = we
            features[1][:pos_onehot.shape[0], :pos_onehot.shape[1]] = pos_onehot
            features[2][:dep_embeddings.shape[1], :dep_embeddings.shape[2]] = dep_embeddings
            features[3][:deps_onehot.shape[0], :deps_onehot.shape[1]] = deps_onehot

            assert features[0].shape == (125, EMBEDDING_DIM)
            assert features[1].shape == (125, EMBEDDING_DIM)
            assert features[2].shape == (125, EMBEDDING_DIM)
            assert features[3].shape == (125, EMBEDDING_DIM)

            ret_dict[i] = SentenceAttributes(*features)

        return ret_dict

    @staticmethod
    def _get_entity_specific_feature(entity, embedding_model, indices, embed_dict):
        features = np.zeros((3, 125, EMBEDDING_DIM))

        we = PreCoParser.get_embedding_for_sent(entity, embedding_model)
        dep_embeddings = embed_dict[indices[0]].dep_embeddings[indices[1]:indices[2]]
        dep_rel = embed_dict[indices[0]].deps_onehot[indices[1]:indices[2]]

        features[0][:we.shape[0], :we.shape[1]] = we
        features[1][:dep_embeddings.shape[0], :dep_embeddings.shape[1]] = dep_embeddings
        features[2][:dep_rel.shape[0], :dep_rel.shape[1]] = dep_rel

        assert features[0].shape == (125, EMBEDDING_DIM)
        assert features[1].shape == (125, EMBEDDING_DIM)
        assert features[2].shape == (125, EMBEDDING_DIM)

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
    def custom_cluster_to_nn_input(sents, clusters):
        """
        Preprocessing steps:
            * Grab entites from sentence using ParseClusters.get_named_entities(sent)
            * Features for NN: Shape: (6, 125, 400, 125) -> (num_features, max_sent_len, EMBEDDING_DIM, max_sent_len)
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

        bar = IncrementalBar('*\tGenerating NN friendly input from custom clusters and senteences', max=len(sents))
        xy_split = []
        xtrain = []
        ytrain = []

        for i in range(len(sents)):
            embed_dict = CoreferenceNetwork._sent_to_we_pos_em(sents[i], embedding_model, pos_onehot_map)
            indiv_features = CoreferenceNetwork._single_custom_cluster_to_nn(
                sents[i], clusters[i], embed_dict, embedding_model)
            sent_vals = list(embed_dict.values())

            xy_split.extend([(TrainingSampleAttributes(*sent_vals[idx], indiv.entity_we, indiv.headword_embed, indiv.headword_relation), cluster)
                             for idx, cluster, indiv in indiv_features])
            bar.next()

        xtrain, ytrain = zip(*xy_split)
        ytrain = CoreferenceNetwork._get_nn_friendly_output(ytrain)
        print(f'*\tTotal training samples: {len(xtrain)}')
        return np.asarray(xtrain), np.asarray(ytrain)
