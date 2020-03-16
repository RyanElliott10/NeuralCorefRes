# -*- coding: utf-8 -*-
# Cluster formatting for neural network input
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import csv
import gc
import itertools
import pprint
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import spacy
from nltk import pos_tag
from progress.bar import IncrementalBar

pretty_printer = pprint.PrettyPrinter()

ClusterIndices = List[List[int]]

REDUCED_SPACY_NE_TAGS = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC',
                         'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME']
REDUCED_SPACY_TAGS = ['NN', 'NNP', 'NNPS', 'NNS', 'PRP', 'PRP$', 'WP']

nlp = spacy.load('en_core_web_sm')


class ParseClusters:
    @staticmethod
    def tokenize_sent(sent: str) -> List[str]:
        doc = nlp(sent)
        return [ent.text for ent in doc]

    @staticmethod
    def _get_all_indices(arr: List[str], entity: str) -> List[Tuple[int, int]]:
        """ Takes a tokenized sentence and multi-word entity and returns the indices that entity appears within the sentence. """
        tokens = [el.text for el in nlp(entity)]
        return [(i, i+len(tokens)) for i in range(len(arr)) if arr[i:i+len(tokens)] == tokens]

    @staticmethod
    def get_named_entities(sent: List[str]) -> DefaultDict[str, List[int]]:
        """ Accepts a tokenized sentence (or sentences) and returns a list of named entities with their indices. """
        if isinstance(sent, str):
            sent = [tok.text for tok in nlp(sent)]
        doc = nlp(' '.join(sent))

        # Get spacy named entities
        ents = defaultdict(str)
        for ent in doc.ents:
            ents[ent.text] = ParseClusters._get_all_indices(sent, ent.text)

        # Get normal nouns, pronouns, its, etc.
        for token in doc:
            if token.tag_ in REDUCED_SPACY_TAGS and token.text not in ents.keys():
                ents[token.text] = ParseClusters._get_all_indices(sent, token.text)

        return ents

    @staticmethod
    def _is_custom_range_in_cluster_range(index_range: List[int], sent_ndx: int, entity_range: List[int]) -> bool:
        return sent_ndx == entity_range[0] and index_range[0] >= entity_range[1] and index_range[1] <= entity_range[2]

    @staticmethod
    def _match_clusters(index_range: List[int], sent_ndx: int, clusters) -> Tuple[List[int], ClusterIndices]:
        """ Iterates through PreCo clusters and returns clusters the index range appears in. """
        inrange_clusters = []
        indicies = []
        for key, cluster in clusters.items():
            for entity_range in cluster:
                if ParseClusters._is_custom_range_in_cluster_range(index_range, sent_ndx, entity_range):
                    indicies.append(key)
                    inrange_clusters.append(entity_range)
        return (indicies, inrange_clusters)

    @staticmethod
    def _get_overlap(index_range: List[int], match: List[int]) -> float:
        """ Returns percentage of the match the word is in. """
        return (index_range[1]-index_range[0]) / (match[2]-match[1])

    @staticmethod
    def _get_best_cluster(index_range: List[int], clusters: List[ClusterIndices], sent_ndx: int) -> List[int]:
        """ Determines which match (cluster) from the PreCo to map the current key to. """
        keys, matches = ParseClusters._match_clusters(index_range, sent_ndx, clusters)
        largest_overlap = (0, None)

        ret_key = None
        for key, match in enumerate(matches):
            overlap = ParseClusters._get_overlap(index_range, match)
            if overlap > largest_overlap[0]:
                largest_overlap = (overlap, match)
                ret_key = keys[key]
        return largest_overlap[1], ret_key

    @staticmethod
    def get_reduced_clusters(sents: List[List[str]], clusters: List[ClusterIndices]) -> DefaultDict[int, ClusterIndices]:
        """ Converts PreCo clusters to my clusters whiling maintaining references. """
        reduced = defaultdict(list)
        [[[reduced[ParseClusters._get_best_cluster(index_range, clusters, sent_ndx)[1]].append((sent_ndx, index_range[0], index_range[1])) for index_range in indices] for (
            key, indices) in ParseClusters.get_named_entities(sent).items()] for (sent_ndx, sent) in enumerate(sents)]

        if None in reduced.keys():
            del reduced[None]
        return reduced

    @staticmethod
    def write_custom_to_file(reductions: Tuple[List[str], List[Dict[int, List[List[int]]]]], filepath: str):
        df = pd.DataFrame.from_records(reductions, columns=[u'sentences', u'mention_clusters'])
        df.to_json(filepath)

    @staticmethod
    def get_from_file(filepath: str) -> Tuple[List[str], List[Dict[int, List[List[int]]]]]:
        df = pd.read_json(filepath, lines=True, encoding='ascii')
        return np.asarray(list(df[u'sentences'][0].values())), np.asarray(list(df[u'mention_clusters'][0].values()))


if __name__ == '__main__':
    sent1 = ['Charlie', 'Schnelz', 'ran', 'to', 'the', 'park', 'and', 'he', ',', 'Charlie', ',', 'had', 'fun', '.', ]
    sent2 = ['The', 'Frank', 'Committee', 'ran', 'to', 'Farrell', 'Smyth', '.']
    sent3 = ['And', 'then', 'he', 'ran', 'to', 'Target', 'across', 'from', 'it', '.']
    sent4 = ['``', 'Is', 'there', 'anything', 'else', 'you', 'need', ',', 'honey', '?', '\'\'']
    sent5 = ['my', 'dad', 'asked', 'me', 'as', 'he', 'put', 'three', 'twenty', 'dollar', 'bills', 'in', 'my', 'hand', '.']

    preco_sents = [sent1, sent2, sent3, sent4, sent5]
    preco_clusters = [
        [[0, 0, 2], [0, 7, 8], [0, 9, 10], [2, 2, 3]],  # Charlie Schnelz, he, Charlie, he
        [[0, 5, 6], [2, 8, 9]],  # park, it
        [[1, 0, 3]],  # The Frank Committee
        [[1, 5, 7]],  # Farrell Smyth
        [[2, 5, 6]],  # Target
        [[0, 2, 6]],  # Random
        [[1, 3, 6]],  # Random
        [[3, 3, 7]],  # anything else you need
        [[3, 5, 6], [3, 8, 9], [4, 0, 1], [4, 3, 4], [4, 12, 13]],   # you, honey, my, me, my
        [[4, 0, 2], [4, 5, 6]],  # my dad,
        [[4, 7, 11]],  # three twenty dollar bills
        [[4, 8, 10]],  # twenty dollar
        [[4, 12, 14]]  # my hand
    ]

    preco_clusters = dict(zip(range(len(preco_clusters)), preco_clusters))
    print(preco_clusters)

    ParseClusters.get_reduced_clusters(preco_sents, preco_clusters)
