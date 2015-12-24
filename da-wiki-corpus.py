#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim
from gensim.corpora import WikiCorpus
import os.path
import logging
import sys

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    
    logger = logging.getLogger(program)
     
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    
    wiki = WikiCorpus("dawiki-20151201-pages-articles.xml.bz2", dictionary = {})
    
    space = " "
    i = 0
    
    output = open("./wiki-corpus.txt", "w")
    
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 5000 == 0):
            logger.info("Saved " + str(i) + " articles")
        #if (i >= 100000):
        #    break
            
    logger.info("Finished processing " + str(i) + " articles")
    
    output.close()

#MmCorpus.serialize("wiki_da_vocab", wiki)