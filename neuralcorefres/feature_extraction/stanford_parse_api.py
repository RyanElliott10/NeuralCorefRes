# -*- coding: utf-8 -*-
# Interface to StanfordCoreNLP
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

from typing import Any, Dict, List, Sequence, Tuple

import nltk
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser

from neuralcorefres.common.parses import Dependency

DependencyGrammar = List[Tuple[Tuple[str, str], str, Tuple[str, str]]]
ConstituencyGrammar = nltk.tree.Tree


class StanfordParseAPI:
    """
    A simple interface to act as a light API between the Stanford Parser and
    my program. Mainly removes redundant code and makes the API easier to use.
    """

    _dependency_parser = CoreNLPDependencyParser(
        url='http://localhost:9000', tagtype='pos')
    _constituency_parser = CoreNLPParser('http://localhost:9000')

    @staticmethod
    def create_dependency_parser(url: str, tagtype: str):
        _dependency_parser = CoreNLPDependencyParser(url=url, tagtype=tagtype)

    @staticmethod
    def create_constituency_parser(url: str):
        _constituency_parser = CoreNLPParser(url)

    @staticmethod
    def _dependency_parse(sent: str) -> List[Dependency]:
        print(sent)
        result = StanfordParseAPI._dependency_parser.raw_parse(sent)
        tmp = list(result.__next__().triples())
        return [Dependency(*res) for res in tmp]

    @staticmethod
    def dependency_parse(sent: str) -> List[Dependency]:
        return StanfordParseAPI._dependency_parse(sent)

    @staticmethod
    def conll_dependency_parse(sent: str) -> Dict[Any, Any]:
        parse, = StanfordParseAPI._dependency_parser.raw_parse(sent)
        return parse.to_conll(4)

    @staticmethod
    def _constituency_parse(sent: str) -> ConstituencyGrammar:
        return StanfordParseAPI._constituency_parser.raw_parse(sent)

    @staticmethod
    def constituency_parse(sent: str) -> ConstituencyGrammar:
        return list(StanfordParseAPI._constituency_parse(sent))[0]

    @staticmethod
    def tags(sent: str) -> List[Tuple[str, str]]:
        return StanfordParseAPI._dependency_parser.tag([sent])
