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
from typing import List

import nltk
import numpy as np
from keras.preprocessing import sequence
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from progress.bar import IncrementalBar

sys.path.append(os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../"))
import neuralcorefres.parsedata.gap_parser as GAPParse
from neuralcorefres.common import Sentence
from neuralcorefres.feature_extraction.gender_classifier import (
    GENDERED_NOUN_PREFIXES, GenderClassifier)
from neuralcorefres.feature_extraction.stanford_parse_api import \
    StanfordParseAPI
from neuralcorefres.model.cluster_network import ClusterNetwork
from neuralcorefres.model.word_embedding import WordEmbedding
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
    embedding_model = WordEmbedding(model_path=".././data/models/word_embeddings/preco-vectors.model")
    print(embedding_model.embedding_model.most_similar(positive=['water', 'sand']))


def preco_parser_demo(data):
    INPUT_MAXLEN = 200
    OUTPUT_MAXLEN = 200
    # embedding_model = WordEmbedding(model_path=".././data/models/word_embeddings/preco-vectors.model")
    embedding_model = WordEmbedding(model_path=".././data/models/word_embeddings/google-vectors.model")
    data = PreCoParser.prep_for_nn(data)
    xtrain, ytrain = PreCoParser.get_train_data(data, INPUT_MAXLEN, OUTPUT_MAXLEN, embedding_model)

    gc.collect()
    np.set_printoptions(threshold=sys.maxsize)

    cluster_network = ClusterNetwork(xtrain[:8000], ytrain[:8000], xtrain[8000:],
                                     ytrain[8000:], inputmaxlen=INPUT_MAXLEN, outputlen=OUTPUT_MAXLEN)
    cluster_network.train()


def train_model():
    data = PreCoParser.get_preco_data(PreCoDataType.TEST)
    preco_parser_demo(data)


def predict_from_model():
    cluster_model = ClusterNetwork()
    cluster_model.load_saved(".././data/models/clusters/small.h5")
    embedding_model = WordEmbedding(model_path=".././data/models/word_embeddings/preco-vectors.model")

    sent = ["``", "Is", "there", "anything", "else", "you", "need", ",", "honey", "?", "''"]
    embeddings = embedding_model.get_embeddings(sent)
    pos_onehot = PreCoParser.get_pos_onehot_for_sent(sent, PreCoParser.get_pos_onehot())
    padded_pos = np.asarray(sequence.pad_sequences([pos_onehot], maxlen=125, dtype='float32'))
    print(cluster_model.predict(embeddings, padded_pos) * len(sent))


if __name__ == "__main__":
    # word_embeddings_demo()
    train_model()
    # predict_from_model()
