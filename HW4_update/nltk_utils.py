import difflib

import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(words), dtype=np.float32)

    for idx, w in enumerate(words):
        for s_word in sentence_words:
            if w == s_word:
                bag[idx] = 1.0
                break
            elif len(w) > 4 and w in s_word:
                bag[idx] = 1.0
                break
            elif difflib.SequenceMatcher(None, w, s_word).ratio() > 0.8:
                bag[idx] = 1.0
                break

    return bag
