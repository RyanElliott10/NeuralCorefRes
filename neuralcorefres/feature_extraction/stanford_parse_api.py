# -*- coding: utf-8 -*-
# Interface to StanfordCoreNLP
#
# Author: Ryan Elliott <ryane.elliott31@gmail.com>
#
# For license information, see LICENSE

from typing import Any, Dict, List, Sequence, Tuple

import nltk
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser

DependencyGrammar = List[Tuple[Tuple[str, str], str, Tuple[str, str]]]
ConstituencyGrammar = nltk.tree.Tree


class StanfordParseAPI:
    """
    A simple interface to act as a light API between the Stanford Parser and
    my program. Mainly removes redundant code and makes the API easier to use.

    To use, run `java -cp "./stanford-corenlp-full-2018-10-05/*" -mx4g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000` in the command line,
    where "./stanford-corenlp-full-2018-10-05/*" is just the path to the given
    directory. Then, create a CoreNLPDependencyParser with the url of http://localhost:9000
    and use it like a normal Python3 object.
    """

    def __init__(self, url: str = 'http://localhost:9000', tagtype: str = 'pos'):
        self._dependency_parser = CoreNLPDependencyParser(
            url=url, tagtype=tagtype)
        self._constituency_parser = CoreNLPParser('http://localhost:9000')

    def _dependency_parse(self, sent: str) -> List[Tuple]:
        result = self._dependency_parser.raw_parse(sent)
        return list(result.__next__().triples())

    def dependency_parse(self, sents: List[str]) -> List[DependencyGrammar]:
        return [self._dependency_parse(sent) for sent in sents]

    def conll_dependency_parse(self, sent: str) -> Dict[Any, Any]:
        parse, = self._dependency_parser.raw_parse(sent)
        return parse.to_conll(4)

    def _constituency_parse(self, sent: str) -> ConstituencyGrammar:
        return self._constituency_parser.raw_parse(sent)

    def constituency_parse(self, sents: List[str]) -> ConstituencyGrammar:
        return [list(self._constituency_parse(sent)) for sent in sents]

    def tags(self, sents: Sequence[str]) -> List[Tuple[str, str]]:
        return self._dependency_parser.tag(sents)
