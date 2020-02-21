import random
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import names


HARD_GENDERED_WORDS = {
    'she': 'female',
    'her': 'female',
    'she\'d': 'female',
    'gal': 'female'
}
NOUN_PREFIXES = set(
    ['NN', 'NNS', 'NNNP', 'NNPS', 'PRP', 'PRP$', 'WP', 'WP$'])
GENDERED_NOUN_PREFIXES = set(['NNNP', 'NNPS', 'PRP', 'PRP$'])


class GenderClassifier:
    def __init__(self):
        self.classifier = self.train_model()

    @staticmethod
    def gender_features(word: str) -> Dict[str, str]:
        return {
            'last_letter': word[-1:],
            'last_two_letters': word[-2:],
            'first_letter': word[0],
            'first_two_letters': word[:2]
        }

    def train_model(self) -> nltk.NaiveBayesClassifier:
        male_labeled_names = [(name, 'male')
                              for name in names.words('male.txt')]
        female_labeled_names = [(name, 'female')
                                for name in names.words('female.txt')]
        labeled_names = male_labeled_names + female_labeled_names
        random.shuffle(labeled_names)

        featuresets = [(GenderClassifier.gender_features(n), gender)
                       for (n, gender) in labeled_names]

        train_set, test_set = featuresets[500:], featuresets[:500]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        return classifier

    def get_gender(self, word: str) -> str:
        if word in HARD_GENDERED_WORDS:
            return HARD_GENDERED_WORDS[word]
        return self.classifier.classify(GenderClassifier.gender_features(word))

    def get_genders(self, sent: str) -> List[Tuple[Tuple[str, str], str]]:
        tokenized = nltk.word_tokenize(sent)
        ret = []
        for word in tokenized:
            ret.append((word, self.get_gender(word[0])))
        return ret

    def get_genders_gen(self, sent: str):
        tokenized = nltk.word_tokenize(sent)
        for word in tokenized:
            yield (word, self.get_gender(word))
