from enum import Enum
from typing import List
import pandas as pd


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


class GAPCoreferenceDatapoint:
    def __init__(self, identifier: str, text: str, pronoun: str, pronoun_offset: int, a: str, a_offset: int, a_coref: bool, b: str, b_offset: int, b_coref: bool):
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


_BASE_FILEPATH = "../data/google_gap-coreference/"
_FILE_TYPES = {
    GAPDataType.TRAIN: "gap-development.tsv",
    GAPDataType.TEST: "gap-test.tsv",
    GAPDataType.VALIDATION: "gap-validation.tsv"
}


def get_GAP_data(data_type: GAPDataType, basepath: str = _BASE_FILEPATH) -> List[GAPCoreferenceDatapoint]:
    full_filepath = basepath + _FILE_TYPES[data_type]
    df = pd.read_csv(full_filepath, sep="\t").drop(["URL"], axis=1)
    data = [GAPCoreferenceDatapoint(*row[1]) for row in df.iterrows()]

    return data
