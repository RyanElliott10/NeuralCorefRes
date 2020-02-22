# -*- coding: utf-8 -*-
# NeuralCorefRes main
#
# Author: Ryan Elliott <ryane.elliott31@gmail.com>
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
