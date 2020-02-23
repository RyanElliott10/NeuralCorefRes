# -*- coding: utf-8 -*-
# NeuralCorefRes main
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

from typing import Tuple


class Dependency:
    def __init__(self, dependent: Tuple[str, str], relation: str, source: Tuple[str, str]):
        self._dependent = dependent
        self._relation = relation
        self._source = source

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