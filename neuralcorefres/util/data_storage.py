# -*- coding: utf-8 -*-
# Local data reader and parser to optimize model training.
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import os
from typing import List

import pandas as pd

from neuralcorefres.feature_extraction.stanford_parse_api import (
    ConstituencyGrammar, DependencyGrammar)
from neuralcorefres.common.parses import Dependency

DATAPATH = os.path.abspath(
    f"{os.path.dirname(os.path.abspath(__file__))}/../../data/local_data")


def write_constituency_file(constitency_parses: ConstituencyGrammar, filename: str = "gap_constituency_parses.csv"):
    print(constitency_parses)


def write_dependency_file(dependency_parses: List[List[DependencyGrammar]], filename: str = "gap_dependency_parses.csv", identifiers: List[str] = None):
    df = None
    for i, dep_parse in enumerate(dependency_parses):
        if identifiers:
            i = identifiers[i]
        deps = [list(parse.raw_data(id=i)) for parse in dep_parse]
        if df is None:
            df = pd.DataFrame(deps, columns=list(
                Dependency.write_format(id=True)))
        else:
            df = df.append(pd.DataFrame(deps, columns=list(
                Dependency.write_format(id=True))), ignore_index=True)
    with open(f"{DATAPATH}/{filename}", "w+") as f:
        f.write(df.to_csv())
