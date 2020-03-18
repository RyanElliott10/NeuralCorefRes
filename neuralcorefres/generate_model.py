# -*- coding: utf-8 -*-
# NeuralCorefRes main
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import argparse
import gc
import os
import pprint
import re
import sys
from itertools import zip_longest
from typing import List

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from progress.bar import IncrementalBar
from tensorflow.keras.preprocessing import sequence

sys.path.append(os.path.abspath(f'{os.path.dirname(os.path.abspath(__file__))}/../'))
import neuralcorefres.parsedata.gap_parser as GAPParse
from neuralcorefres.common import Sentence
from neuralcorefres.feature_extraction.gender_classifier import (
    GENDERED_NOUN_PREFIXES, GenderClassifier)
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


def train_model(samples: int):
    def generate_data():
        """ For batching data when training on entire dataset. """
        while True:
            for i in range(35):
                for j in range(1, 11):
                    sents, clusters = ParseClusters.get_from_file(f'../data/PreCo_1.0/custom_dps/train_b{i}.json')
                    x_train, y_train = CoreferenceNetwork.custom_cluster_to_nn_input(
                        sents[(j-1)*100:j*100], clusters[(j-1)*100:j*100])
                    yield x_train, y_train

        print(x_train.shape, y_train.shape)

    INPUT_MAXLEN = 128
    OUTPUT_MAXLEN = 128

    sents, clusters = ParseClusters.get_from_file('../data/PreCo_1.0/custom_dps/train_b0.json')
    x_train, y_train = CoreferenceNetwork.custom_cluster_to_nn_input(sents[:samples], clusters[:samples])

    print('\n * x_train, y_train shape before:', x_train.shape, y_train.shape)
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2], x_train.shape[1], x_train.shape[3]))
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2], x_train.shape[3], x_train.shape[1]))
    # print('\n * x_train, y_train shape after:', x_train.shape, y_train.shape)

    coreference_network = CoreferenceNetwork(x_train[:int(len(x_train)*0.9)], y_train[:int(len(x_train)*0.9)], x_train[int(len(x_train)*0.9):],
                                             y_train[int(len(x_train)*0.9):], inputmaxlen=INPUT_MAXLEN, outputlen=OUTPUT_MAXLEN)
    eval = coreference_network.train()
    print(eval)


def predict_from_model(sent: str = None):
    sent = 'Charlie ran to the park where he proceeded to meet a new friend.'

    coreference_network = CoreferenceNetwork()
    preds = coreference_network.predict(sent)

    print(sent)
    pprint.pprint(preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict', type=str, help='Sentence to predict on')
    parser.add_argument('-t', '--train', action='store_true', help='Train a model')
    parser.add_argument('--samples', type=int, default=sys.maxsize, help='Limit the training samples')
    args = parser.parse_args()
    if args.train:
        train_model(args.samples)
    elif args.predict:
        predict_from_model(args.predict)
