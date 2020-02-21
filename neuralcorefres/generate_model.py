# -*- coding: utf-8 -*-
# NeuralCorefRes main
#
# Author: Ryan Elliott <ryane.elliott31@gmail.com>
#
# For license information, see LICENSE

import pprint
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.wsd import lesk

import parsedata.gap_parser as GAPParse
from feature_extraction.gender_classifier import (GENDERED_NOUN_PREFIXES,
                                                  GenderClassifier)
from feature_extraction.stanford_parse_api import StanfordParseAPI

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


def gender_demo(sent):
    classifier = GenderClassifier()

    tagged = nltk.pos_tag(nltk.word_tokenize(sent))
    for word in tagged:
        if word[1] in GENDERED_NOUN_PREFIXES:
            print(word, classifier.get_gender(word[0]))
    print(classifier.get_gender('marine'))


if __name__ == "__main__":
    data: List[GAPParse.GAPCoreferenceDatapoint] = GAPParse.get_GAP_data(
        GAPParse.GAPDataType.TRAIN)

    print(data[0].text)
    sparser = StanfordParseAPI()
    deps = sparser.dependency_parse([d.text for d in data[:50]])
    pretty_printer.pprint(deps[0])

    const_parse = sparser.constituency_parse([d.text for d in data[:2]])
    for el in const_parse:
        print(el)
