# -*- coding: utf-8 -*-
# NeuralCorefRes main
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import os
import pprint
import sys
import re
from typing import List

import nltk
from nltk.corpus import stopwords
from progress.bar import IncrementalBar

sys.path.append(os.path.abspath(
    f"{os.path.dirname(os.path.abspath(__file__))}/../"))
from neuralcorefres.util.word_embedding import *
from neuralcorefres.util.preprocess import single_output
from neuralcorefres.feature_extraction.util import findall_entities, spacy_entities
from neuralcorefres.util.data_storage import (write_constituency_file,
                                              write_dependency_file)
from neuralcorefres.feature_extraction.stanford_parse_api import \
    StanfordParseAPI
from neuralcorefres.feature_extraction.gender_classifier import (
    GENDERED_NOUN_PREFIXES, GenderClassifier)
from neuralcorefres.common import Sentence
import neuralcorefres.parsedata.gap_parser as GAPParse
import neuralcorefres.parsedata.preco_parser as PreCoParser


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
    sents: List[GAPParse.GAPCoreferenceDatapoint] = GAPParse.get_gap_data(
        GAPParse.GAPDataType.TRAIN, class_type=Sentence)

    bar = IncrementalBar('Parsing Sentences...', max=len(sents))
    for sent in sents:
        sent.parse()
        bar.next()

    write_dependency_file([sent._dep_parse for sent in sents], identifiers=[
                          sent._id for sent in sents])


def word_embeddings():
    """ Deprecated. Use the PreCo dataset. """
    sents = GAPParse.get_gap_data(
        [GAPParse.GAPDataType.TRAIN, GAPParse.GAPDataType.VALIDATION], class_type=Sentence)
    model = WordEmbedding(
        model_path='.././data/models/word_embeddings/google-vectors.model', sents=sents)

    texts = [sent.alphanumeric_text for sent in sents]
    nid = []
    total_tokens = []
    for text in texts:
        tokenized = word_tokenize(text)
        for i, token in enumerate(tokenized):
            if not model.embedding_model.__contains__(token):
                embedding = model.estimate_embedding(tokenized[i-5:i+5], token)
                print(
                    f'{token}: {model.embedding_model.similar_by_vector(embedding, topn=1)}')
    nid = set(nid)


def word_embeddings_demo():
    """ Demo of word embeddings using a pre-trained model on PreCo data. """
    embedding_model = WordEmbedding(model_path=".././data/models/word_embeddings/preco-vectors.model")
    print(embedding_model.embedding_model.most_similar(positive=['california', 'sand']))

def preco_parser_demo(data):
    data = PreCoParser.prep_for_nn(data)

    print("\n\n")
    for key, value in data.items():
        print("KEY:", key)
        [print(f"\t{c.__str__()}") for c in value]


if __name__ == "__main__":
    data = PreCoParser.get_preco_data(PreCoParser.PreCoDataType.TRAIN)
    word_embeddings_demo()
    preco_parser_demo(data)
