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

sys.path.append(os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../"))
import neuralcorefres.parsedata.gap_parser as GAPParse
from neuralcorefres.common import Sentence
from neuralcorefres.feature_extraction.gender_classifier import (
    GENDERED_NOUN_PREFIXES, GenderClassifier)
from neuralcorefres.feature_extraction.stanford_parse_api import \
    StanfordParseAPI
from neuralcorefres.util.data_storage import (write_constituency_file,
                                              write_dependency_file)
from neuralcorefres.feature_extraction.util import findall_entities, spacy_entities


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


def yeet():
    sents: List[GAPParse.GAPCoreferenceDatapoint] = GAPParse.get_GAP_data(
        GAPParse.GAPDataType.TRAIN, class_type=Sentence)

    bar = IncrementalBar('Parsing Sentences...', max=len(sents))
    for sent in sents:
        sent.parse()
        bar.next()

    write_dependency_file([sent._dep_parse for sent in sents], identifiers=[
                          sent._id for sent in sents])


if __name__ == "__main__":
    sent = u"Bobby Tarantino ran to the bench. He then sat down on it."
    con = [StanfordParseAPI.constituency_parse(sent)]
    dep = [StanfordParseAPI.dependency_parse(sent)] * 5

    tagged = StanfordParseAPI.tags(sent)
    print(f"{tagged}\n")
    print(findall_entities(tagged))
