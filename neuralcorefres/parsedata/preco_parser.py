from collections import namedtuple
from enum import Enum
from typing import List, Tuple

import pandas as pd
from progress.bar import IncrementalBar

Cluster = List[str]
ClusterIndicies = namedtuple('ClusterIndicies', 'sent_idx begin_idx end_idx')


class PreCoDataType(Enum):
    TRAIN = 0
    TEST = 1


class EntityCluster:
    def __init__(self, entities: Cluster, indices: ClusterIndicies):
        self.entities = entities
        self.indices = ClusterIndicies(*indices)

    def __str__(self):
        return f"{self.entities} | {self.indices}"


class PreCoCoreferenceDatapoint:
    def __init__(self, id, sents: List[Cluster], entity_clusters: EntityCluster):
        self.id = id
        self.sents = sents
        self.entity_clusters = self._get_sorted_clusters(entity_clusters)

    def _get_sorted_clusters(self, clusters) -> List[EntityCluster]:
        return sorted(clusters, key=lambda cluster: cluster.indices.sent_idx)

    @staticmethod
    def parse_entity_clusters(sentences: List[List[str]], entity_clusters: List[List[List[int]]]):
        """
        Per the PreCo website, mention clusters are in the following form:
        [ [ [ sentence_idx, begin_idx, end_idx ] ] ]

        Where the end index is one past the last word in the cluster, and all
        indicies are zero-based.

        Example:

        Sentences:
        [
            [ "Charlie", "had", "fun", "at", "the", "park", "." ],
            [ "He", "slid", "down", "the", "slide", "." ]
        ]
        Mention Clusters:
        [
            [ [0, 0, 1], [1, 0, 1] ],   // Charlie, he
            [ [0, 5, 6] ],              // park
            [ [1, 4, 5] ]               // slide
        ]
        """
        clusters = [[EntityCluster(sentences[sent_idx][begin_idx:end_idx], (sent_idx, begin_idx, end_idx))
                     for sent_idx, begin_idx, end_idx in cluster][0] for cluster in entity_clusters]
        return clusters


_BASE_FILEPATH = "../data/PreCo_1.0/"
_FILE_TYPES = {
    PreCoDataType.TRAIN: "train.json",
    PreCoDataType.TEST: "dev.json"
}


def get_preco_data(data_type: PreCoDataType, basepath: str = _BASE_FILEPATH, class_type: PreCoCoreferenceDatapoint = PreCoCoreferenceDatapoint) -> List[PreCoCoreferenceDatapoint]:
    ret_lst = []
    full_filepath = basepath + _FILE_TYPES[data_type]
    df = pd.read_json(full_filepath, lines=True)
    bar = IncrementalBar(
        'Reading and creating objects from PreCo dataset...', max=len(df))
    for index, el in df.iterrows():
        entity_clusters = PreCoCoreferenceDatapoint.parse_entity_clusters(
            el['sentences'], el['mention_clusters'])
        ret_lst.append(PreCoCoreferenceDatapoint(
            el['id'], el['sentences'], entity_clusters))
        bar.next()
    return ret_lst


def main():
    get_preco_data(PreCoDataType.TRAIN)


if __name__ == "__main__":
    main()
