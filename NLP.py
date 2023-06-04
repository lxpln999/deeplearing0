import sys
import os
import numpy as np
pypath = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../common')

def preprocess(text):
    text = text.lower()
    text = text.replace(',', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


sys.path.append('..')
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print('corpus:', corpus)
print('word_to_id:', word_to_id)
print('id_to_word:', id_to_word)

C = np.array([[0, 1, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 1, 1, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 1, 0]], dtype=np.int32)
print('C[0]:',C[0])

import sys
sys.path.append(pypath)
from common import create_co_matrix,cos_similarity