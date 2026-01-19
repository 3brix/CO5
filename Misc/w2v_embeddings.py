# -*- coding: utf-8 -*-
"""
Task W2V Embeddings Demo

Created on Mon November 24 16:23:08 2025

@author: agha
"""

import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

nltk.download('brown', quiet=True)

random.seed(42)

sentences = [[w.lower() for w in sent] for sent in brown.sents()]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=24,
    sg=1
)

top_n = 200
words = list(model.wv.index_to_key)[:top_n]
word_vectors = model.wv[words]

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
embeddings_2d = tsne.fit_transform(word_vectors)


plt.figure(figsize=(12, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.8)

for i, word in enumerate(words):
    plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
                 xytext=(5, 5), textcoords='offset points', fontsize=15)

plt.title("t-SNE visualization of Brown Word2Vec embeddings")
plt.show()


