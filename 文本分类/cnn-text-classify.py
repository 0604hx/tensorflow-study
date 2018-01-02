"""
源码来自：https://github.com/dmesquita/understanding_tensorflow_nn.git
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

my_graph = tf.Graph()
with tf.Session(graph=my_graph) as sess:
    x = tf.constant([1, 3, 6])
    y = tf.constant([2, 3, 4])
    op = tf.add(x, y)
    result = sess.run(fetches=op)
    print(result)

vocab = Counter()

text = "Hi from Brazil"

for word in text.split(' '):
    word_lowercase = word.lower()
    vocab[word_lowercase] += 1


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i

    return word2index


word2index = get_word_2_index(vocab)

total_words = len(vocab)
matrix = np.zeros((total_words), dtype=float)

for word in text.split():
    matrix[word2index[word.lower()]] += 1

print("Hi from Brazil:", matrix)

matrix = np.zeros((total_words), dtype=float)
text = "Hi"
for word in text.split():
    matrix[word2index[word.lower()]] += 1

print("Hi:", matrix)

"""
Building the neural network
"""
categories = ["comp.graphics", "sci.space", "rec.sport.baseball"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

print('total texts in train:', len(newsgroups_train.data))
print('total texts in test:', len(newsgroups_test.data))

print('text[0]=', newsgroups_train.data[0])
print('category:', newsgroups_train.target[0])

vocab = Counter()

for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

total_words = len(vocab)

print("Total words:", total_words)


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index


word2index = get_word_2_index(vocab)

print("Index of the word 'the':", word2index['the'])
