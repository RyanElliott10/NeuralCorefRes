# -*- coding: utf-8 -*-
# Parser for PreCo dataset
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

from collections import defaultdict, namedtuple
from enum import Enum
from itertools import chain
from typing import DefaultDict, List, Tuple

import pandas as pd
from progress.bar import IncrementalBar

from multiprocessing import Pool

from neuralcorefres.model.word_embedding import WordEmbedding

Cluster = List[str]
Tensor = List[float]

ClusterIndicies = namedtuple('ClusterIndicies', 'sent_idx begin_idx end_idx')
ClusteredSentence = namedtuple('ClusteredSentence', 'sentence clusters')
ClusteredDictKey = namedtuple('ClusteredDictKey', 'id sentence_index sentence')


class PreCoDataType(Enum):
    TRAIN = 0
    TEST = 1


class EntityCluster:
    def __init__(self, entity: Cluster, indices: ClusterIndicies):
        self.entity = entity
        self.indices = ClusterIndicies(*indices)

    def __str__(self):
        return f"{self.entity} | {self.indices}"


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

    def __str__(self):
        sub_strs = '\t' + '\n\t'.join([str(cluster)
                                       for cluster in self.entity_clusters])
        return f"{self.id}\n{sub_strs}"


_BASE_FILEPATH = "../data/PreCo_1.0/"
_FILE_TYPES = {
    PreCoDataType.TRAIN: "train.json",
    PreCoDataType.TEST: "dev.json"
}


class PreCoParser:
    @staticmethod
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

    @staticmethod
    def flatten_tokenized(sents: List[PreCoCoreferenceDatapoint]):
        """ Flattens tokenized lists of PreCo datapoints. """
        all_sents = [sent.sentences for sent in sents]
        return [tokens for sentences in all_sents for tokens in sentences]

    @staticmethod
    def prep_for_nn(preco_data: List[PreCoCoreferenceDatapoint]) -> DefaultDict[str, List[EntityCluster]]:
        """
        Returns a dictionary with key: ClusteredDictKey(example_id, sent_idx) and value:
        list of entity clusters for the given sentence with the example.

        Example using __str__ on EntityCluster for visualization purposes:
        {
            dev_00001_0: [
                ['anything', 'else', 'you', 'need'] | ClusterIndicies(sent_idx=0, begin_idx=3, end_idx=7),
            ],
            dev_00001_1:  [
                ['three', 'twenty', 'dollar', 'bills'] | ClusterIndicies(sent_idx=1, begin_idx=7, end_idx=11)
                ['twenty', 'dollar'] | ClusterIndicies(sent_idx=1, begin_idx=8, end_idx=10)
                ['my', 'hand'] | ClusterIndicies(sent_idx=1, begin_idx=12, end_idx=14)
            ]
        }
        """
        organized_data = defaultdict(list)
        for dp in preco_data:
            [organized_data[ClusteredDictKey(dp.id, cluster.indices.sent_idx, tuple(
                dp.sents[cluster.indices.sent_idx]))].append(cluster) for cluster in dp.entity_clusters]
        return organized_data

    @staticmethod
    def get_train_data(data: List[PreCoCoreferenceDatapoint]) -> Tuple[List[Tensor], List[Tensor]]:
        embedding_model = WordEmbedding(
            model_path=".././data/models/word_embeddings/preco-vectors.model")

        xtrain = []
        ytrain = []
        bar = IncrementalBar(
            'Parsing data into xtrain, ytrain', max=len(data))
        for key, value in data.items():
            sentence_embeddings = embedding_model.get_embeddings(key.sentence)

            cluster_indices = list(
                sum([cluster.indices for cluster in value], ()))
            # Delete every third element to remove sentence index
            del cluster_indices[0::3]

            # TODO move this into the second step of the network pipeline
            # Cluster embeddings shouldn't be a part of the training data when predicting clusters. SHould be used when pushing clusters through coreference NN
            # cluster_embeddings = list(chain(*[embedding_model.get_embeddings(
            #     cluster.entity) for cluster in value]))
            # xtrain.append(sentence_embeddings + cluster_embeddings)
            xtrain.append(sentence_embeddings)
            ytrain.append(cluster_indices)
            bar.next()

        print(np.asarray(xtrain))
        return (np.asarray(xtrain), np.asarray(ytrain))

    @staticmethod
    def check_data(data):
        """ Checks data for any ill-formatted datapoints. """
        pool = Pool(os.cpu_count()-1)
        pool.map(PreCoParser._check_data, data)

    @staticmethod
    def _check_data(data):
        embedding_model = WordEmbedding(
            model_path=".././data/models/word_embeddings/preco-vectors.model")
        [embedding_model.embedding_model.most_similar(
            positive=[d], topn=1) for d in data]


def main():
    data = PreCoParser.prep_for_nn(data)
    xtrain, ytrain = PreCoParser.get_train_data(data)


if __name__ == "__main__":
    main()
