import pprint
import nltk
from nltk.wsd import lesk
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPDependencyParser

from feature_extraction.stanford_parser import StanfordParser
from feature_extraction.gender_classifier import GenderClassifier

pretty_printer = pprint.PrettyPrinter()

REMOVED_STOPWORDS = set(['my', 'he', 'you\'ll', 'her', 'i', 'hers', 'who', 'your',
                         'himself', 'yourself', 'own', 'you\'re', 'you\'d', 'we',
                         'myself', 'yourselves', 'yours', 'ours', 'she', 'she\'s',
                         'his', 'you\'ve', 'me', 'they', 'him', 'whom', 'them',
                         'their', 'theirs', 'herself', 'themselves', 'you',
                         'ourselves', 'itself', 'our', 'this', 'that', 'those'])
STOPWORDS = set.difference(set(stopwords.words('english')), REMOVED_STOPWORDS)


def stanford():
    sent_one = u'I shot an elephant in my pajamas.'
    sent_two = u'I don\'t like when you do that.'
    sent_three = u'My neck is itchy.'
    sent_four = u'I\'m super excited to play racquetball because I\'m going to beat your ass.'
    sent_five = u'The lawyer questioned the witness.'

    sent_four = ' '.join(
        [w.lower() for w in sent_four.split() if w.lower() not in STOPWORDS])

    print(f"{sent_five}\n")
    sparser = StanfordParser()
    dependencies = sparser.dependency_grammars_lst(
        [sent_one, sent_two, sent_three, sent_four, sent_five])
    print("Dependencies:")
    pretty_printer.pprint(dependencies)

    tags = sparser.tags(sent_five)
    print("\nTags:")
    pretty_printer.pprint(tags)

    conll = sparser.conll_parse(sent_five)
    print(f"\nCONLL parse:\n{conll}")

    word = 'witness'
    wsd = lesk(nltk.word_tokenize(sent_five), word)
    print(f"{word}: {wsd.definition()}")


if __name__ == "__main__":
    stanford()
