# -*- coding: utf-8 -*-
# Word embedding utils
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import os
from typing import List, Sequence

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize

from neuralcorefres.common.sentence import Sentence

EMBEDDING_DIM = 400


def keras_impl(sents: List[Sentence]):
    texts = [sent._text for sent in sents]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    print(sequences)

    data = pad_sequences(sequences)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print(data)


def tokenize_text(sents: List[Sentence]) -> List[List[str]]:
    return [word_tokenize(sent._text) for sent in sents]


def gensim_impl(model_path: str, sents: List[Sentence] = None):
    if os.path.exists(model_path):
        print('Loading model')
        model = KeyedVectors.load(model_path)
        word_vectors = model
    elif sents is None:
        print('Loading Google News word vectors...')
        model = KeyedVectors.load_word2vec_format(
            '.././data/GoogleNews-vectors-negative300.bin.gz', binary=True)
        model.wv.save(model_path)
        word_vectors = model
    else:
        print('Training model on new data...')
        model = Word2Vec(sentences=tokenize_text(sents),
                         size=EMBEDDING_DIM, min_count=0, workers=5)
        word_vectors = model.wv
        word_vectors.save('.././data/models/word_embeddings/gap-vectors.model')

    words = list(word_vectors.vocab)
    print(word_vectors.most_similar('cute'))
    print(word_vectors.most_similar_cosmul(
        positive=['England', 'America'], negative=['London']))


def embedding_tensor(model_path: str = None, sents: List[Sentence] = None) -> Sequence:
    gensim_impl(model_path, sents)
