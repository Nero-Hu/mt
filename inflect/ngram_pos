#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import sys
from collections import defaultdict
from itertools import izip

def utf8read(f):
    return codecs.open(f, 'r', 'utf-8')

class Ngram_pos:
    """
    ngram model with pos tags
    """
    def __init__(self, formFile, lemmaFile, posFile, ngram):
        self.gramStats(formFile, lemmaFile, posFile, ngram)
    
    def gramStats(self, formFile, lemmaFile, posFile, ngram):
        # List of dictionaries that contains the stats for ngram mapping
        self.lemmaStats = [defaultdict(lambda : defaultdict(lambda : defaultdict(float)))] * ngram
        
        for words, lemmas, tags in izip(utf8read(formFile),utf8read(lemmaFile), utf8read(posFile)):
            # Collect bigram with based on lemma
            words = words.rstrip().lower().split()
            lemmas = lemmas.rstrip().lower().split()
            tags = tags.split()
            # Collect stats for n-grams joint with pos
            for i in xrange(1, ngram + 1):
                for n in range(len(lemmas) + 1 - i):
                    self.lemmaStats[i-1][tuple(lemmas[n:n+i])][tuple(tags[n:n+i])][tuple(words[n:n+i])] += 1

        return self.lemmaStats

    def getUnigram(self, posDict):
        """
        return dict with key to be the words, by summming up all possible pos
        """
        (max_count, best_unigram) = (0,0)
        unigramDict = defaultdict(float)
        for posTag, wordDict in posDict.iteritems():
            for word, counts in wordDict.iteritems():
                unigramDict[word] += counts
                if unigramDict[word] > max_count:
                    max_count = unigramDict[word]
                    best_unigram = word

        return best_unigram

    def backoffLM(self, lemma, tags):
        # See if ngram exists, if not backoff to n-1 gram
        if tuple(lemma) not in self.lemmaStats[len(lemma)-1]:
            if len(lemma) == 1:
                return lemma
            return self.backoffLM(lemma[0:len(lemma)-1], tags[0:len(tags)-1])
        elif tuple(tags) not in self.lemmaStats[len(lemma)-1][tuple(lemma)]:
            if len(lemma) == 1:
                #return largest unigram prob
                return self.getUnigram(self.lemmaStats[len(lemma)-1][tuple(lemma)])
            else:
                return self.backoffLM(lemma[0:len(lemma)-1], tags[0:len(tags)-1])
        # goes here just return the largest
        return sorted(self.lemmaStats[len(lemma)-1][tuple(lemma)][tuple(tags)].iteritems(),\
         key=lambda (k,v): v,reverse=True)[0][0]
            

def main():
    PARSER = argparse.ArgumentParser(description="Inflect a lemmatized corpus with n-gram model")
    PARSER.add_argument("-t", type=str, default="data/train", help="training data prefix")
    PARSER.add_argument("-l", type=str, default="lemma", help="lemma file suffix")
    PARSER.add_argument("-w", type=str, default="form", help="word file suffix")
    PARSER.add_argument("-n", type=int, default=3, help="n-gram model(default 3 to be trigram)")
    PARSER.add_argument("-d", type=str, default="data/dtest", help="test file")
    args = PARSER.parse_args()

    # Python sucks at UTF-8
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout) 

    ng_pos = Ngram_pos(args.t + "." + args.w, args.t + "." + args.l, args.t + ".tag", args.n)

    for lemmas, tags in izip(utf8read(args.d + "." + args.l), utf8read(args.d + ".tag")):
        lemmas = lemmas.rstrip().lower().split()
        tags = tags.split()
        sen = ""
        i = 0
        while i < len(lemmas):
            lem = ng_pos.backoffLM(lemmas[i:min(i+args.n, len(lemmas))], tags[i:min(i+args.n, len(tags))])
            i += len(lem)
            for le in lem:
                sen += le + " "
        print sen.rstrip()

if __name__ == '__main__':
    main()