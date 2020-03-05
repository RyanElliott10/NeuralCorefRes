from typing import List
from enum import Enum

import pandas as pd


class PreCoDataType(Enum):
    TRAIN = 0
    TEST = 1


class PreCoCoreferenceDatapoint:
    def __init__(self, id, sentences, mention_clusters):
        self.id = id
        # every sentence is tokenized
        self.sentences: List[List[str]] = sentences
        self.mention_clusters = mention_clusters

    @staticmethod
    def parse_mention_clusters(sentences: List[List[str]], mention_clusters: List[List[List[int]]]):
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
        [[[0, 1, 2], []]]
        """
        clusters = [[sentences[sent_idx][begin_idx:end_idx] for sent_idx,
                     begin_idx, end_idx in cluster] for cluster in mention_clusters]
        # print(clusters)


_BASE_FILEPATH = "../data/PreCo_1.0/"
_FILE_TYPES = {
    PreCoDataType.TRAIN: "train.json",
    PreCoDataType.TEST: "dev.json",
}


def get_preco_data(data_type: PreCoDataType, basepath: str = _BASE_FILEPATH, class_type: PreCoCoreferenceDatapoint = PreCoCoreferenceDatapoint) -> List[PreCoCoreferenceDatapoint]:
    ret_lst = []
    full_filepath = basepath + _FILE_TYPES[data_type]
    df = pd.read_json(full_filepath, lines=True)
    for index, el in df.iterrows():
        mention_clusters = PreCoCoreferenceDatapoint.parse_mention_clusters(
            el['sentences'], el['mention_clusters'])
        ret_lst.append(PreCoCoreferenceDatapoint(
            el['id'], el['sentences'], el['mention_clusters']))
    return ret_lst


def main():
    get_preco_data([PreCoDataType.TEST])


if __name__ == "__main__":
    main()
