# -*- coding: utf-8 -*-
# Parser for Google's GAP-Coreference dataset
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

from enum import Enum
from typing import List
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd


REMOVED_STOPWORDS = set(['my', 'he', 'you\'ll', 'her', 'i', 'hers', 'who', 'your',
                         'himself', 'yourself', 'own', 'you\'re', 'you\'d', 'we',
                         'myself', 'yourselves', 'yours', 'ours', 'she', 'she\'s',
                         'his', 'you\'ve', 'me', 'they', 'him', 'whom', 'them',
                         'their', 'theirs', 'herself', 'themselves', 'you',
                         'ourselves', 'itself', 'our', 'this', 'that', 'those'])
STOPWORDS = set.difference(set(stopwords.words('english')), REMOVED_STOPWORDS)


"""
This dataset is far from ideal; it only contains very few pronouns (he, her,
his, she, him, etc.) and contains fairly complex sentences despite the low
pronoun variety. Additionally, the pronouns being referred to are strictly
upercase, properly formatted names (Cheryl Cassidy, MacKenzie, etc.), so any
model trained on this likely wouldn't be able to be used elsewhere.
"""


class GAPDataType(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


_BASE_FILEPATH = "../data/google_gap-coreference/"
_FILE_TYPES = {
    GAPDataType.TRAIN: "gap-development.tsv",
    GAPDataType.TEST: "gap-test.tsv",
    GAPDataType.VALIDATION: "gap-validation.tsv"
}


class GAPCoreferenceDatapoint:
    def __init__(self, identifier: str, text: str, pronoun: str, pronoun_offset: int, a: str, a_offset: int, a_coref: bool, b: str, b_offset: int, b_coref: bool):
        self._id = identifier
        self.text = text
        self._pronoun = pronoun
        self._pronoun_offset = pronoun_offset
        self._a = a
        self._a_offset = a_offset
        self._a_coref = a_coref
        self._b = b
        self._b_offset = b_offset
        self._b_coref = b_coref
        self.input_ready_alpha()

    def input_ready_alpha(self):
        """ Removes all non-letter from raw texts, tokenizes, and removes modified stopwords. """
        self.alphanumeric_text = [w for w in word_tokenize(re.sub(
            r'[^A-Za-z ]+', '', self.text, flags=re.UNICODE)) if w not in STOPWORDS]


def get_gap_data(data_type: List[GAPDataType], basepath: str = _BASE_FILEPATH, class_type: GAPCoreferenceDatapoint = GAPCoreferenceDatapoint) -> List[GAPCoreferenceDatapoint]:
    ret_lst = []
    for datat in data_type:
        full_filepath = basepath + _FILE_TYPES[datat]
        df = pd.read_csv(full_filepath, sep="\t").drop(["URL"], axis=1)
        ret_lst = ret_lst + [class_type(*row[1]) for row in df.iterrows()]
    return ret_lst
