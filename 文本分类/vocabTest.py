import unittest
from collections import Counter
import numpy as np


def word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i
    return word2index


class VocabTest(unittest.TestCase):

    def test_word_to_index(self):
        texts = ["Apache Tomcat v8.5 JDK8", "Nginx Web Server", "Jetty Web Server", "JBoss Server"]
        words = []
        for text in texts:
            words.extend(text.strip().split(" "))
        print(words)

        vocab = Counter(words)
        print(vocab)
        indexes = word_2_index(vocab)
        print(indexes)

        for text in texts:
            matrix = np.zeros((len(indexes)), dtype=float)

            for w in text.split():
                matrix[indexes[w]] += 1
            print(matrix, text)



if __name__ == '__main__':
    unittest.main()
