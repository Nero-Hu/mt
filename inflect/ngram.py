#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import sys
from collections import defaultdict
from itertools import izip

def utf8read(f):
    return codecs.open(f, 'r', 'utf-8')
class Ngram:
    """
    Simple ngram model
    """
    def __init__(self, formFile, lemmaFile, ngram):
        self.gramStats(formFile, lemmaFile, ngram)
        
    def gramStats(self, formFile, lemmaFile, ngram):
        # List of dictionaries that contains the stats for ngram mapping
        self.lemmaStats = [defaultdict(lambda : defaultdict(float))] * ngram
        
        for words, lemmas in izip(utf8read(formFile),utf8read(lemmaFile)):
            # Collect bigram with based on lemma
            words = words.rstrip().lower().split()
            lemmas = lemmas.rstrip().lower().split()
            # Collect stats for n-grams
            for i in xrange(1, ngram + 1):
                for n in range(len(lemmas) + 1 - i):
                    self.lemmaStats[i-1][tuple(lemmas[n:n+i])][tuple(words[n:n+i])] += 1

        return self.lemmaStats

    def backoffLM(self, lemma):
        if tuple(lemma) not in self.lemmaStats[len(lemma)-1]:
            if len(lemma) == 1:
                return lemma
            return self.backoffLM(lemma[0:len(lemma)-1])
        return sorted(self.lemmaStats[len(lemma)-1][tuple(lemma)].iteritems(), key=lambda (k,v): v,reverse=True)[0][0]

def main():
    PARSER = argparse.ArgumentParser(description="Inflect a lemmatized corpus with n-gram model")
    PARSER.add_argument("-t", type=str, default="data/train", help="training data prefix")
    PARSER.add_argument("-l", type=str, default="lemma", help="lemma file suffix")
    PARSER.add_argument("-w", type=str, default="form", help="word file suffix")
    PARSER.add_argument("-n", type=int, default=2, help="n-gram model(default 2 to be bigram)")
    PARSER.add_argument("-d", type=str, default="data/dtest", help="test file")
    args = PARSER.parse_args()

    # Python sucks at UTF-8
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout) 

    ng = Ngram(args.t + "." + args.w, args.t + "." + args.l, args.n)

    for line in utf8read(args.d + "." + args.l):
        line = line.rstrip().lower().split()
        sen = ""
        i = 0
        while i < len(line):
            lem = ng.backoffLM(line[i:min(i+args.n, len(line))])
            i += len(lem)
            for le in lem:
                sen += le + " "
        print sen.rstrip()

if __name__ == '__main__':
    main()