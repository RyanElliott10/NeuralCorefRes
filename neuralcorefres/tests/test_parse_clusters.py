import unittest

from neuralcorefres.parsedata.parse_clusters import ParseClusters


class TestParseClustesr(unittest.TestCase):
    def test_get_named_entities(self):
        sent1 = ["Charlie", "Schnelz", "ran", "to", "the", "park", "and", "he", ",", "Charlie", ",", "had", "fun", ".", ]
        self.assertEqual(ParseClusters.get_named_entities(sent1)['Charlie'], [(0, 1), (9, 10)])
        self.assertEqual(ParseClusters.get_named_entities(sent1)['Charlie Schnelz'], [(0, 2)])
