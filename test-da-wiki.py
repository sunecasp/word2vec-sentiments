#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models import LsiModel

import logging
import os.path
import sys
import cPickle as pickle

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

model = Doc2Vec.load('./da-wiki.d2v')
model.init_sims(replace = True)

model.most_similar(positive=u"hillerød århus".split(), negative=u"lyngby".split())

graph = TSNE(n_components = 2, random_state = 0)
words = u"god dårlig positiv negativ".split()
#words = u"mand kvinde dreng pige".split()
X = model[words]
Y = graph.fit_transform(X)

fig = plt.figure(figsize=(12,6))
plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
plt.title("t-SNE")

for word, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(
        word,
        xy = (x, y), xytext = (-10, 10),
        textcoords = 'offset points')

plt.show()