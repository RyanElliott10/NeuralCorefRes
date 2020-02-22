# -*- coding: utf-8 -*-
# NeuralCorefRes main
#
# Author: Ryan Elliott <ryane.elliott31@gmail.com>
#
# For license information, see LICENSE

import os
import pprint
import sys
from typing import List
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

import nltk
from nltk.corpus import stopwords

import neuralcorefres.parsedata.gap_parser as GAPParse
from neuralcorefres.common import Sentence
from neuralcorefres.feature_extraction.gender_classifier import (
    GENDERED_NOUN_PREFIXES, GenderClassifier)
from neuralcorefres.feature_extraction.stanford_parse_api import \
    StanfordParseAPI


pretty_printer = pprint.PrettyPrinter()


"""
TODO: Parse the data and store the features of each sentence in tsv files to
avoid absurd parsing (and therefore training) times.
"""

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


if __name__ == "__main__":
    con = StanfordParseAPI.constituency_parse("Bobby ran to the park.")
    dep = StanfordParseAPI.dependency_parse("Bobby ran to the park.")
    for d in dep:
        print(d.reveal())

    sents: List[GAPParse.GAPCoreferenceDatapoint] = GAPParse.get_GAP_data(
        GAPParse.GAPDataType.TRAIN, class_type=Sentence)
    
    for sent in sents[:1]:
        sent.parse()
