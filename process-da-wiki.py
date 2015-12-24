#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys
import cPickle as pickle

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences
        
    def bigrams(self):
        self.phrases = []
        self.tags = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.phrases.append(utils.to_unicode(line).split())
                    self.tags.append([prefix + '_%s' % item_no])
        self.transform = gensim.models.Phrases(self.phrases)

        self.sentences = []        
        for phrase, tag in zip(self.phrases, self.tags):
            self.sentences.append(LabeledSentence(self.transform[phrase], tag))
            
        del self.phrases, self.tags
        
        return self.sentences
        
sources = {"wiki-corpus.txt":"UNSUP"}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=12, size=800, sample=1e-4, negative=5, workers=2)

model.build_vocab(sentences.to_array())
#model.build_vocab(sentences.bigrams())

for epoch in range(10):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())
    
model.save('./da-wiki.d2v')
#model = Doc2Vec.load('./da-wiki.d2v')

#model.init_sims(replace = True)

#model.most_similar(positive=u"hillerød århus".split(), negative=u"lyngby".split())