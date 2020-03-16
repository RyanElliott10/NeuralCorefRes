# -*- coding: utf-8 -*-
# NeuralCorefRes main
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import gc
import os
import pprint
import re
import sys
from itertools import zip_longest
from typing import List

import nltk
import numpy as np
from keras.preprocessing import sequence
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from progress.bar import IncrementalBar

sys.path.append(os.path.abspath(f'{os.path.dirname(os.path.abspath(__file__))}/../'))
import neuralcorefres.parsedata.gap_parser as GAPParse
from neuralcorefres.common import Sentence
from neuralcorefres.feature_extraction.gender_classifier import (
    GENDERED_NOUN_PREFIXES, GenderClassifier)
from neuralcorefres.feature_extraction.stanford_parse_api import \
    StanfordParseAPI
from neuralcorefres.model.cluster_network import ClusterNetwork
from neuralcorefres.model.coreference_network import CoreferenceNetwork
from neuralcorefres.model.word_embedding import WordEmbedding
from neuralcorefres.parsedata.parse_clusters import ParseClusters
from neuralcorefres.parsedata.preco_parser import PreCoDataType, PreCoParser
from neuralcorefres.util.data_storage import write_dependency_file


pretty_printer = pprint.PrettyPrinter()


REMOVED_STOPWORDS = set(['my', 'he', 'you\'ll', 'her', 'i', 'hers', 'who', 'your',
                         'himself', 'yourself', 'own', 'you\'re', 'you\'d', 'we',
                         'myself', 'yourselves', 'yours', 'ours', 'she', 'she\'s',
                         'his', 'you\'ve', 'me', 'they', 'him', 'whom', 'them',
                         'their', 'theirs', 'herself', 'themselves', 'you',
                         'ourselves', 'itself', 'our', 'this', 'that', 'those'])
STOPWORDS = set.difference(set(stopwords.words('english')), REMOVED_STOPWORDS)


def gender_demo(sent: str):
    classifier = GenderClassifier()

    tagged = nltk.pos_tag(nltk.word_tokenize(sent))
    for word in tagged:
        if word[1] in GENDERED_NOUN_PREFIXES:
            print(word, classifier.get_gender(word[0]))
    print(classifier.get_gender('marine'))


def write_deps():
    sents: List[GAPParse.GAPCoreferenceDatapoint] = GAPParse.get_gap_data(GAPParse.GAPDataType.TRAIN, class_type=Sentence)
    bar = IncrementalBar('Parsing Sentences...', max=len(sents))
    for sent in sents:
        sent.parse()
        bar.next()

    write_dependency_file([sent._dep_parse for sent in sents], identifiers=[sent._id for sent in sents])


def word_embeddings():
    """ Deprecated. Use the PreCo dataset. """
    sents = GAPParse.get_gap_data([GAPParse.GAPDataType.TRAIN, GAPParse.GAPDataType.VALIDATION], class_type=Sentence)
    model = WordEmbedding(model_path='.././data/models/word_embeddings/google-vectors.model', sents=sents)

    texts = [sent.alphanumeric_text for sent in sents]
    nid = []
    total_tokens = []
    for text in texts:
        tokenized = word_tokenize(text)
        for i, token in enumerate(tokenized):
            if not model.embedding_model.__contains__(token):
                embedding = model.estimate_embedding(tokenized[i-5:i+5], token)
                print(f'{token}: {model.embedding_model.similar_by_vector(embedding, topn=1)}')
    nid = set(nid)


def word_embeddings_demo():
    """ Demo of word embeddings using a pre-trained model on PreCo data. """
    embedding_model = WordEmbedding(model_path='.././data/models/word_embeddings/preco-vectors.model')
    print(embedding_model.embedding_model.most_similar(positive=['water', 'sand']))


def preco_parser_demo(data):
    INPUT_MAXLEN = 200
    OUTPUT_MAXLEN = 200
    # embedding_model = WordEmbedding(model_path='.././data/models/word_embeddings/preco-vectors.model')
    embedding_model = WordEmbedding(model_path='.././data/models/word_embeddings/google-vectors.model')
    data = PreCoParser.prep_for_nn(data)
    x_train, y_train = PreCoParser.get_train_data(data, INPUT_MAXLEN, OUTPUT_MAXLEN, embedding_model)

    gc.collect()
    np.set_printoptions(threshold=sys.maxsize)

    # cluster_network = ClusterNetwork(x_train[:8000], y_train[:8000], x_train[8000:],
    #                                  y_train[8000:], inputmaxlen=INPUT_MAXLEN, outputlen=OUTPUT_MAXLEN)
    cluster_network = ClusterNetwork(x_train[:190], y_train[:190], x_train[190:],
                                     y_train[190:], inputmaxlen=INPUT_MAXLEN, outputlen=OUTPUT_MAXLEN)
    cluster_network.train()


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def parse_clusters():
    data = PreCoParser.get_preco_data(PreCoDataType.TRAIN)[4000:]
    total_clusters = dict(zip(range(len(data[1])), data[1]))

    print()
    bar = IncrementalBar('*\tConverting PreCo dataset to custom dataset form', max=len(data))
    reductions: List[Tuple[List[str], Dict[int, List[List[int]]]]] = []

    batches = grouper(data, 1000)
    for i, batch in enumerate(batches):
        reductions: List[Tuple[List[str], Dict[int, List[List[int]]]]] = []
        for dp in batch:
            reductions.append((dp[0], ParseClusters.get_reduced_clusters(dp[0], dict(zip(range(len(dp[1])), dp[1])))))
            bar.next()
        ParseClusters.write_custom_to_file(reductions, f'../data/PreCo_1.0/custom_dps/train_b{i+4}.json')


def train_model():
    sents, clusters = ParseClusters.get_from_file('../data/PreCo_1.0/custom_dps/dev.json')
    x_train, y_train = CoreferenceNetwork.custom_cluster_to_nn_input(sents[:100], clusters[:100])

    print(x_train[40][0].shape)
    print(x_train[40][1].shape)
    print(x_train[40][2].shape)
    print(x_train[40][3].shape)
    print(x_train[40][4].shape)
    print(x_train[40][5].shape)
    print(x_train[40][6].shape)
    print(x_train.shape, y_train.shape)

    INPUT_MAXLEN = 125
    OUTPUT_MAXLEN = 125
    coreference_network = CoreferenceNetwork(x_train[:int(len(x_train)*0.9)], y_train[:int(len(x_train)*0.9)], x_train[int(len(x_train)*0.9):],
                                             y_train[int(len(x_train)*0.9):], inputmaxlen=INPUT_MAXLEN, outputlen=OUTPUT_MAXLEN)
    coreference_network.train()


def predict_from_model():
    sent = 'Charlie Schnlez ran to the park where he had fun.'
    sent = 'Sara walked around town and she saw Target.'
    sent = 'They say that sticks and stones may break your bones, but will never hurt you.'

    coreference_network = CoreferenceNetwork()
    coreference_network.predict(sent)


if __name__ == '__main__':
    train_model()
    # predict_from_model()
