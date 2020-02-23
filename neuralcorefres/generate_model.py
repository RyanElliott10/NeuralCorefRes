# -*- coding: utf-8 -*-
# NeuralCorefRes main
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
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
from neuralcorefres.util.data_storage import write_constituency_file, write_dependency_file

from progress.bar import IncrementalBar


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
    # con = [StanfordParseAPI.constituency_parse("Bobby ran to the bench. He then sat down on it.")]
    # dep = [StanfordParseAPI.dependency_parse("Bobby ran to the bench. He then sat down on it.")] * 5
    # # for d in dep:
    # #     print(d.reveal())
    # # print(con)

    # # write_constituency_file(con)
    # write_dependency_file(dep, identifiers=[0, 1, 2, 3, 4])

    sents: List[GAPParse.GAPCoreferenceDatapoint] = GAPParse.get_GAP_data(
        GAPParse.GAPDataType.TRAIN, class_type=Sentence)
    
    bar = IncrementalBar('Parsing Sentences...', max=len(sents))
    for sent in sents:
        sent.parse()
        bar.next()

    
    write_dependency_file([sent._dep_parse for sent in sents], identifiers=[sent._id for sent in sents])

    # print(sents[0]._text)
    # for dep in sents[0]._dep_parse:
    #     print(dep.reveal())