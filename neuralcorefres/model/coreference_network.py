import pprint
import sys
from collections import defaultdict, namedtuple
from typing import DefaultDict, List, Tuple

import numpy as np
from keras import Sequential
from keras.layers import (LSTM, Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
                          TimeDistributed)
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing import sequence
from progress.bar import IncrementalBar

from neuralcorefres.model.word_embedding import WordEmbedding
from neuralcorefres.parsedata.preco_parser import PreCoParser

np.set_printoptions(threshold=sys.maxsize)
pretty_printer = pprint.PrettyPrinter()

Tensor = List[float]
SentenceAttributes = namedtuple('SentenceAttributes', ['we', 'pos_onehot', 'dep_embeddings', 'deps_onehot'])
EntityAtttibutes = namedtuple('EntityAtttibutes', ['entity_we', 'headword_embed', 'headword_relation'])
TrainingSampleAttributes = namedtuple('TrainingSampleAttributes', ['we', 'pos_onehot', 'dep_embeddings', 'deps_onehot', 'entity_we', 'headword_embed',
                                                                   'headword_relation'])


class CoreferenceNetwork:
    def __init__(self, xtrain, ytrain, xtest, ytest):
        pass

    def _build_model(self):
        pass

    def train(self):
        pass

    def predict(self, pred_data):
        pass

    @staticmethod
    def _pad_1d_tensor(pos_onehot, maxlen: int):
        return sequence.pad_sequences(pos_onehot, maxlen, dtype='float32', padding='post')

    @staticmethod
    def _sent_to_we_pos_em(sents: List[List[str]], embedding_model, pos_onehot_map) -> DefaultDict[int, SentenceAttributes]:
        """ Returns a dictionary with key as index (maybe not safe? perhaps use ' '.join(sent)) containing word embeddings.  """
        ret_dict = defaultdict(list)
        for i, sent in enumerate(sents):
            we = PreCoParser.get_embedding_for_sent(sent, embedding_model)
            pos_onehot = CoreferenceNetwork._pad_1d_tensor(PreCoParser.get_pos_onehot_map_for_sent(sent, pos_onehot_map), we.shape[1])
            dep_embeddings = PreCoParser.get_dep_embeddings(sent, embedding_model)  # Horribly slow
            deps_onehot = CoreferenceNetwork._pad_1d_tensor(PreCoParser.get_deps_onehot(sent), we.shape[1])  # Horribly slow

            ret_dict[i] = SentenceAttributes(we, pos_onehot, dep_embeddings, deps_onehot)
        return ret_dict

    @staticmethod
    def _get_entity_specific_feature(entity, embedding_model, indices, embed_dict):
        we = PreCoParser.get_embedding_for_sent(entity, embedding_model)
        dep_embedding = embed_dict[indices[0]].dep_embeddings[indices[1]:indices[2]]
        dep_rel = embed_dict[indices[0]].deps_onehot[indices[1]:indices[2]]

        return EntityAtttibutes(we, dep_embedding, dep_rel)

    @staticmethod
    def _single_custom_cluster_to_nn(sents, clusters, embed_dict, embedding_model):
        """
        Iterate through clusters and get the word embeddings for each cluster as shape:
            (clusters_len, num_in_cluster, EMBEDDONG_DIM)
            [
                [
                    [], [], [], []
                ]
            ]
        """
        features = []
        for key, cluster in clusters.items():
            for indices in cluster:
                sub = sents[indices[0]][indices[1]:indices[2]]
                features.append((indices[0], CoreferenceNetwork._get_entity_specific_feature(sub, embedding_model, indices, embed_dict)))

        return features

        # print('\n\n')

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

        def create(sent_vals, i, *args):
            print(f'\ni: {i}')
            return TrainingSampleAttributes(sent_vals[i], args)

        def tmp(ret, index):
            print(f'\nINDEX BABY: {index}')
            return ret

        embedding_model = WordEmbedding(model_path='.././data/models/word_embeddings/google-vectors.model')
        pos_onehot_map = PreCoParser.get_pos_onehot_map()

        bar = IncrementalBar('*\tGenerating NN friendly input from custom clusters and senteences', max=len(sents))
        xtrain = []
        ytrain = []

        for i in range(len(sents)):
            embed_dict = CoreferenceNetwork._sent_to_we_pos_em(sents[i], embedding_model, pos_onehot_map)
            indiv_features = CoreferenceNetwork._single_custom_cluster_to_nn(sents[i], clusters[i], embed_dict, embedding_model)
            sent_vals = list(embed_dict.values())

            xtrain.extend([TrainingSampleAttributes(*sent_vals[idx], indiv.entity_we, indiv.headword_embed, indiv.headword_relation)
                           for idx, indiv in indiv_features])
            [ytrain.extend(arr) for arr in list(clusters[i].values())]
            bar.next()

        print(f'\n{len(xtrain)} {len(ytrain)}')
        return np.asarray(xtrain), np.asarray(ytrain)
