# -*- coding: utf-8 -*-
# General utils
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import re
from typing import List, Tuple

import spacy

TaggedPOS = Tuple[str, str]

DIRECT_POS = ["NN"]
INDIRECT_POS = ["RP", "WP", "PRP"]


def findall_entities(tagged_sent: List[TaggedPOS]) -> Tuple[List[Tuple[TaggedPOS, int]], List[Tuple[TaggedPOS, int]]]:
    """
    Finds all direct and indirect nouns/pronouns in a POS tagged sentence.
    """
    directs = []
    indirects = []
    for i, tag in enumerate(tagged_sent):
        if len(re.findall(r"(?=("+'|'.join(DIRECT_POS)+r"))", tag[1])) > 0:
            directs.append((tag, i))
        elif len(re.findall(r"(?=("+'|'.join(INDIRECT_POS)+r"))", tag[1])) > 0:
            indirects.append((tag, i))

    return (directs, indirects)


def spacy_entities(sent: str) -> List:
    """
    Less accurate than custom finall_entities, but here regardless.
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sent)
    return [entity.text for entity in doc.ents]
