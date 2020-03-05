# -*- coding: utf-8 -*-
# Word embedding utils
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import os
import time
from typing import List, Sequence

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from progress.bar import IncrementalBar

from neuralcorefres.common.sentence import Sentence
from neuralcorefres.parsedata.preco_parser import PreCoCoreferenceDatapoint

Tensor = List[float]


EMBEDDING_DIM = 400


class WordEmbedding:
    def __init__(self, model_path: str = None, sents: List[Sentence] = None, is_tokenized: bool = False):
        self.embedding_model = self._load_model(
            model_path, sents, is_tokenized)

    def _load_model(self, model_path: str, sents: List[Sentence], is_tokenized: bool):
        if model_path is not None and os.path.exists(model_path):
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
            if is_tokenized:
                model = Word2Vec(
                    sentences=sents, size=EMBEDDING_DIM, workers=5)
            else:
                model = Word2Vec(sentences=tokenizetext(sents),
                                 size=EMBEDDING_DIM, min_count=0, workers=5)
            word_vectors = model.wv
            word_vectors.save(
                '.././data/models/word_embeddings/gap-vectors.model')

        return word_vectors

    @staticmethod
    def flatten_tokenized(sents: List[PreCoCoreferenceDatapoint]):
        """ Flattens tokenized lists of PreCo datapoints. """
        all_sents = [sent.sentences for sent in sents]
        return [tokens for sentences in all_sents for tokens in sentences]

    def save_current_model(self, filepath: str):
        self.embedding_model.save(filepath)

    def keras_impl(self, sents: List[Sentence]):
        texts = [sent.alphanumeric_text for sent in sents]
        tokenizer = Tokenizer()
        tokenizer.fit_ontexts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        print(sequences)

        data = pad_sequences(sequences)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        labels = to_categorical(np.asarray(labels))
        print('Shape of data tensor:', data.shape)
        print(data)

    def _estimate_embedding(self, surrounding_words: List[str], unknown: str) -> List[float]:
        """
        Used to determine an estimated word embedding for a given word. Takes
        average embeddings of the surrounding words.
        """
        summations = None
        valid_num = 0
        for surr in surrounding_words:
            if self.embedding_model.__contains__(surr):
                valid_num += 1
                arr = np.asarray(self.embedding_model[surr])
                if summations is None:
                    summations = arr
                else:
                    summations = np.add(summations, arr)
        if summations is not None:
            return self.embedding_model.similar_by_vector(summations, topn=1)
        return None

    def tokenizetext(self, sents: List[Sentence]) -> List[List[str]]:
        return [word_tokenize(sent.alphanumeric_text) for sent in sents]

    def _get_embedding(self, tokenized: str):
        """ FIXME this takes a long time to run. """
        embeddings = []
        for i, token in enumerate(tokenized):
            if not self.embedding_model.__contains__(token):
                # FIXME this takes a long time. Can be optimized by storing array of indices that need to be fixed and fixing at end using current embeddings
                embedding = self._estimate_embedding(tokenized[i-3:i+3], token)
                if embedding is not None:
                    embeddings.append(embedding)
            else:
                embeddings.append(self.embedding_model[token])
        print(embeddings)
        return embeddings

    def get_embeddings(self, sents: List[Sentence]) -> List[Tensor]:
        """
        Gets embeddings for the input sentences. Supports GapCoreferenceDatapoint
        """
        embeddings = []
        bar = IncrementalBar('Generating word embeddings...', max=len(sents))
        for i, sent in enumerate(sents):
            embeddings.append(self._get_embedding(sent.alphanumeric_text))
            bar.next()
        return embeddings
