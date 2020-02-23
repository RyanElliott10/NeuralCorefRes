
# -*- coding: utf-8 -*-
# Interface to StanfordCoreNLP
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

from typing import List

import nltk

from neuralcorefres.common.parses import Dependency
from neuralcorefres.parsedata.gap_parser import GAPCoreferenceDatapoint


class Sentence(GAPCoreferenceDatapoint):
    def __init__(self, *args):
        super().__init__(*args)
        self._dep_parse: List[Dependency] = []
        self._const_parse: nltk.tree.Tree = None

    def parse(self):
        """
        Generates constituency/dependency parses, etc. prepped for feature extraction.
        """
        from neuralcorefres.feature_extraction.stanford_parse_api import StanfordParseAPI

        self._dep_parse = StanfordParseAPI.dependency_parse(self._text)
        self._const_parse = StanfordParseAPI.constituency_parse(self._text)

    def extract_features(self):
        """
        Prep for model training, construct dict for model input.
        """
        pass

    def extract_labels(self):
        """
        Used for training, creates dict with expected model output.
        """
        pass
