# -*- coding: utf-8 -*-
# Parser for PreCo dataset
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import gc
from collections import defaultdict, namedtuple
from enum import Enum
from itertools import chain
from multiprocessing import Pool
from typing import DefaultDict, List, Tuple

import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.utils import to_categorical
from nltk import pos_tag
from nltk.data import load
from progress.bar import IncrementalBar
import spacy

from neuralcorefres.model.word_embedding import EMBEDDING_DIM
from neuralcorefres.feature_extraction.stanford_parse_api import StanfordParseAPI

Cluster = List[str]
Tensor = List[float]

ClusterIndicies = namedtuple('ClusterIndicies', 'sent_idx begin_idx end_idx')
ClusteredSentence = namedtuple('ClusteredSentence', 'sentence clusters')
ClusteredDictKey = namedtuple('ClusteredDictKey', 'id sentence_index sentence')

SPACY_DEP_TAGS = ['acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'cop', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl',
                  'intj', 'mark', 'meta', 'neg', 'nn', 'nounmod', 'npmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'obj', 'obl', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'root', 'xcomp']

nlp = spacy.load('en_core_web_sm')
POS_ONE_HOT_LEN = 45


class PreCoDataType(Enum):
    TRAIN = 0
    TEST = 1


class EntityCluster:
    def __init__(self, entity: Cluster, indices: ClusterIndicies):
        self.entity = entity
        self.indices = ClusterIndicies(*indices)

    def __str__(self):
        return f'{self.entity} | {self.indices}'


class PreCoCoreferenceDatapoint:
    def __init__(self, id, sents: List[Cluster], sorted_entity_clusters: EntityCluster):
        self.id = id
        self.sents = sents
        self.sorted_entity_clusters = self._get_sorted_clusters(sorted_entity_clusters)

    def _get_sorted_clusters(self, clusters) -> List[EntityCluster]:
        return sorted(clusters, key=lambda cluster: cluster.indices.sent_idx)

    @staticmethod
    def parse_sorted_entity_clusters(sentences: List[List[str]], sorted_entity_clusters: List[List[List[int]]]):
        """
        Per the PreCo website, mention clusters are in the following form:
        [ [ [ sentence_idx, begin_idx, end_idx ] ] ]

        Where the end index is one past the last word in the cluster, and all
        indicies are zero-based.

        Example:

        Sentences:
        [
            [ 'Charlie', 'had', 'fun', 'at', 'the', 'park', '.' ],
            [ 'He', 'slid', 'down', 'the', 'slide', '.' ]
        ]
        Mention Clusters:
        [
            [ [0, 0, 1], [1, 0, 1] ],   // Charlie, he
            [ [0, 5, 6] ],              // park
            [ [1, 4, 5] ]               // slide
        ]
        """
        clusters = [[EntityCluster(sentences[sent_idx][begin_idx:end_idx], (sent_idx, begin_idx, end_idx))
                     for sent_idx, begin_idx, end_idx in cluster][0] for cluster in sorted_entity_clusters]
        return clusters

    def __str__(self):
        sub_strs = '\t' + '\n\t'.join([str(cluster) for cluster in self.sorted_entity_clusters])
        return f'{self.id}\n{sub_strs}'


_BASE_FILEPATH = '../data/PreCo_1.0/'
_FILE_TYPES = {
    PreCoDataType.TRAIN: 'train.json',
    PreCoDataType.TEST: 'dev.json'
}


class PreCoParser:
    @staticmethod
    def get_pos_onehot_map():
        return pd.get_dummies(list(load('help/tagsets/upenn_tagset.pickle').keys()))

    @staticmethod
    def get_spacy_deps_onehot():
        return pd.get_dummies(SPACY_DEP_TAGS)

    @staticmethod
    def get_preco_data(data_type: PreCoDataType, basepath: str = _BASE_FILEPATH, class_type: PreCoCoreferenceDatapoint = PreCoCoreferenceDatapoint):
        ret_lst = []
        full_filepath = basepath + _FILE_TYPES[data_type]
        df = pd.read_json(full_filepath, lines=True, encoding='ascii')

        bar = IncrementalBar('*\tReading and creating objects from PreCo dataset', max=len(df))
        for index, el in df.iterrows():
            ret_lst.append((el[u'sentences'], el[u'mention_clusters']))
            bar.next()

        gc.collect()
        return ret_lst

    @staticmethod
    def flatten_tokenized(sents: List[PreCoCoreferenceDatapoint]):
        """ Flattens tokenized lists of PreCo datapoints. """
        return [tokens for sentences in [sent.sentences for sent in sents] for tokens in sentences]

    @staticmethod
    def get_embedding_for_sent(sent: List[str], embedding_model) -> List[Tensor]:
        """ Get embeddings as array of embeddings. """
        return embedding_model.get_embeddings(sent)

    @staticmethod
    def get_pos_onehot_map_for_sent(sent: List[str], pos_onehot) -> List[Tensor]:
        """ Get POS as array of one-hot arrays. In same order as words from sentence (if used correctly). """
        return np.asarray([pos_onehot[p].to_numpy() if p in pos_onehot.keys() else np.zeros(len(pos_onehot.keys())) for p in list(zip(*pos_tag(sent)))[1]])

    @staticmethod
    def get_dep_embeddings(sent: List[str], embedding_model) -> List[Tensor]:
        doc = spacy.tokens.doc.Doc(nlp.vocab, words=sent)
        for name, proc in nlp.pipeline:
            doc = proc(doc)

        assert len(doc) == len(sent)
        sorted_deps = [token.head.text for token in doc]
        return PreCoParser.get_embedding_for_sent(sorted_deps, embedding_model)

    @staticmethod
    def get_dep_distances(sent: List[str]):
        doc = spacy.tokens.doc.Doc(nlp.vocab, words=sent)
        for name, proc in nlp.pipeline:
            doc = proc(doc)

        assert len(doc) == len(sent)
        sorted_deps = [token.head.text for token in doc]
        print(sorted_deps)

    @staticmethod
    def get_deps_onehot(sent: List[str]):
        deps_onehot = PreCoParser.get_spacy_deps_onehot()
        doc = spacy.tokens.doc.Doc(nlp.vocab, words=sent)
        for name, proc in nlp.pipeline:
            doc = proc(doc)

        assert len(doc) == len(sent)
        return np.asarray([deps_onehot[p].to_numpy() if p in deps_onehot.keys() else np.zeros(len(deps_onehot.keys())) for p in [token.dep_ for token in doc]])

    @staticmethod
    def pad_1d_tensor(t, maxlen=EMBEDDING_DIM):
        return sequence.pad_sequences([t], maxlen=EMBEDDING_DIM, dtype='float32', padding='post')[0]

    @staticmethod
    def _invalid_data(sentence_embeddings: Tensor, dep_embeddings: Tensor, curr_sent: List[str], maxinputlen: int, maxoutputlen: int) -> bool:
        return sentence_embeddings.shape[0] == 0 or sentence_embeddings.shape[0] != len(curr_sent) \
            or dep_embeddings.shape != sentence_embeddings.shape or len(curr_sent) > maxinputlen or len(dep_embeddings) > maxinputlen

    @staticmethod
    def prep_for_nn(preco_data: List[PreCoCoreferenceDatapoint]) -> DefaultDict[str, List[EntityCluster]]:
        """
        Returns a dictionary with key: ClusteredDictKey(example_id, sent_idx) and value:
        list of entity clusters for the given sentence with the example.

        Example using __str__ on EntityCluster for visualization purposes:
        {
            dev_00001_0: [
                ['anything', 'else', 'you', 'need'] | ClusterIndicies(sent_idx=0, begin_idx=3, end_idx=7),
            ],
            dev_00001_1:  [
                ['three', 'twenty', 'dollar', 'bills'] | ClusterIndicies(sent_idx=1, begin_idx=7, end_idx=11)
                ['twenty', 'dollar'] | ClusterIndicies(sent_idx=1, begin_idx=8, end_idx=10)
                ['my', 'hand'] | ClusterIndicies(sent_idx=1, begin_idx=12, end_idx=14)
            ]
        }
        """
        organized_data = defaultdict(list)
        for dp in preco_data:
            [organized_data[ClusteredDictKey(dp.id, cluster.indices.sent_idx, tuple(
                dp.sents[cluster.indices.sent_idx]))].append(cluster) for cluster in dp.sorted_entity_clusters]

        return organized_data

    @staticmethod
    def get_train_data(data: DefaultDict[ClusteredDictKey, PreCoCoreferenceDatapoint], maxinputlen: int, maxoutputlen: int, embedding_model) -> Tuple[List[Tensor], List[Tensor]]:
        """
        (n_samples, n_words, n_attributes (word embedding, pos, etc))
        [ [ [ word_embedding, pos ] ] ]

        xtrain[sentence_sample][word_position][attribute]
        xtrain[37][5] -> sixth word's attributes in 38th sentence (np.ndarray containing two np.ndarrays)
        xtrain[0][0][0] -> word_embedding (np.ndarray)
        xtrain[0][0][1] -> pos one-hot encoding (np.ndarray)
        """
        xtrain = np.empty((len(data), maxinputlen, 4, EMBEDDING_DIM))
        ytrain = []
        pos_onehot = PreCoParser.get_pos_onehot_map()
        deps_onehot = PreCoParser.get_spacy_deps_onehot()

        bar = IncrementalBar('*\tParsing data into xtrain, ytrain', max=len(data))
        for sent_ndx, (key, value) in enumerate(data.items()):
            curr_sent = key.sentence

            sentence_embeddings = PreCoParser.get_embedding_for_sent(curr_sent, embedding_model)
            sent_pos = PreCoParser.get_pos_onehot_map_for_sent(curr_sent, pos_onehot)
            dep_embeddings = PreCoParser.get_dep_embeddings(curr_sent, embedding_model)
            deps = PreCoParser.get_deps_onehot(curr_sent)

            if PreCoParser._invalid_data(sentence_embeddings, dep_embeddings, curr_sent, maxinputlen, maxoutputlen):
                # Unusable data
                bar.next()
                continue

            assert len(curr_sent) == sentence_embeddings.shape[0]
            assert sentence_embeddings.shape == dep_embeddings.shape
            assert deps.shape[0] == dep_embeddings.shape[0]
            assert sentence_embeddings.shape[0] == sent_pos.shape[0]

            for word_ndx in range(len(sentence_embeddings)):
                xtrain[sent_ndx][word_ndx][0] = sentence_embeddings[word_ndx]
                xtrain[sent_ndx][word_ndx][1] = dep_embeddings[word_ndx]
                xtrain[sent_ndx][word_ndx][2] = PreCoParser.pad_1d_tensor(sent_pos[word_ndx])
                xtrain[sent_ndx][word_ndx][3] = PreCoParser.pad_1d_tensor(deps[word_ndx])

            cluster_indices = list(sum([cluster.indices for cluster in value], ()))
            # Delete every third element to remove sentence index
            del cluster_indices[0::3]
            assert len(cluster_indices) % 2 == 0

            cluster_indices = sequence.pad_sequences([cluster_indices], maxlen=maxoutputlen, dtype='float32', padding='post')[0]
            assert cluster_indices.shape == (maxoutputlen,)

            ytrain.append(np.asarray(cluster_indices) / len(curr_sent))
            bar.next()

        gc.collect()
        ytrain = np.asarray(ytrain, dtype='float32')
        assert ytrain[0].shape == (maxoutputlen,)

        return (xtrain, ytrain)


def main():
    data = PreCoParser.prep_for_nn(data)
    xtrain, ytrain = PreCoParser.get_train_data(data)


if __name__ == '__main__':
    main()
