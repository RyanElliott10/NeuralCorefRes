from typing import List
from enum import Enum


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


class GapCoreferenceDatapoint:
    def __init__(self, identifier: str, text: str, pronoun: str, pronoun_offset: int, a: str, a_offset: int, a_coref: bool, b: str, b_offset: int, b_coref: bool, url: str):
        self.id = identifier
        self.text = text
        self.pronoun = pronoun
        self.pronoun_offset = pronoun_offset
        self.a = a
        self.a_offset = a_offset
        self.a_coref = a_coref
        self.b = b
        self.b_offset = b_offset
        self.b_coref = b_coref


BASE_FILEPATH = "../../data/google_gap-coreference/"
FILE_TYPES = {
    GAPDataType.TRAIN: "gap-development.tsv",
    GAPDataType.TEST: "gap-test.tsv",
    GAPDataType.VALIDATION: "gap-validation.tsv"
}


def get_GAP_data(data_type: GAPDataType, basepath: str = BASE_FILEPATH) -> List[GapCoreferenceDatapoint]:
    full_filepath = basepath + FILE_TYPES[data_type]
    data: List[GapCoreferenceDatapoint] = []
    with open(full_filepath, "r") as f:
        for line in f:
            delimited = line.split('\t')
            if (delimited[2] == "Pronoun"):
                continue
            data.append(GapCoreferenceDatapoint(*delimited))

    return data
