# -*- coding: utf-8 -*-
# Classes used to interpret Stanford's CoreNLP server output.
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

from typing import Tuple


class Dependency:
    def __init__(self, source: Tuple[str, str], relation: str, dependent: Tuple[str, str],):
        self._source = source
        self._relation = relation
        self._dependent = dependent

    def reveal(self) -> str:
        return f"DEP: {self._dependent} REL: {self._relation} SOURCE: {self._source}"

    def raw_data(self, id: int = None) -> Tuple[str, str, str]:
        if id is None:
            return (self._dependent, self._relation, self._source)
        return (id, self._dependent, self._relation, self._source)

    @staticmethod
    def write_format(id: bool = False) -> str:
        if not id:
            return ("dependent", "relation", "source")
        return ("id", "dependent", "relation", "source")

    def __str__(self) -> str:
        return f"{self._source} -> {self._dependent} | {self._relation}"