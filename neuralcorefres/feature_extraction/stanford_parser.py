from typing import List, Tuple, Optional
from nltk.parse.corenlp import CoreNLPDependencyParser


DependencyGrammar = List[Tuple[Tuple[str, str], str, Tuple[str, str]]]


class StanfordParser:
    """
    A simple interface to act as a light API between the Stanford Parser and
    my program. Mainly removes redundant code and makes the API easier to use.

    To use, run `java -cp "./stanford-corenlp-full-2018-10-05/*" -mx4g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000` in the command line,
    where "./stanford-corenlp-full-2018-10-05/*" is just the path to the given
    directory. Then, create a CoreNLPDependencyParser with the url of http://localhost:9000
    and use it like a normal Python3 object.
    """

    def __init__(self, url: Optional[str] = 'http://localhost:9000', tagtype: Optional[str] = 'pos'):
        self.dependency_parser = CoreNLPDependencyParser(
            url=url, tagtype=tagtype)

    def dependency_grammars(self, sent: str) -> List[Tuple]:
        result = self.dependency_parser.raw_parse(sent)
        return list(result.__next__().triples())

    def dependency_grammars_lst(self, sents: str) -> List:
        results: List[DependencyGrammar] = [
            self.dependency_grammars(sent) for sent in sents]
        print("RESULTS:", results[0][0])
        return results

    def conll_parse(self, sent: str) -> dict:
        parse, = self.dependency_parser.raw_parse(sent)
        return parse.to_conll(4)

    def tags(self, sents: List) -> List:
        if not isinstance(sents, list):
            sents = [sents]
        return self.dependency_parser.tag(sents)
