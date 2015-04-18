#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import sys
from collections import defaultdict
from itertools import izip
from tree import DepTree

def utf8read(f):
    return codecs.open(f, 'r', 'utf-8')

class BiDep:
    """Bigram dependency tree
    """
    def __init__(self, formFile, lemmaFile, posFile, treeFile):
        self.gramStats(formFile, lemmaFile, posFile, treeFile)

    def gramStats(self, formFile, lemmaFile, posFile, treeFile):
        """
        Collect stats for grams with dependency
        """
        self.depStats = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        self.lemmaStats = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        
        for words, lemmas, tags, trees in izip(utf8read(formFile),\
            utf8read(lemmaFile),utf8read(posFile), utf8read(treeFile)):
            # Collect bigram with based on lemma
            words = words.rstrip().lower().split()
            lemmas = lemmas.rstrip().lower().split()
            tags = tags.split()
            tree = DepTree(trees)
            # Collect stats for n-grams joint with pos
            for i,node in enumerate(tree):
                self.depStats[(lemmas[node.parent_index()-1],lemmas[i])]\
                [(tree.node(node.parent_index()).label(), node.label())][words[i]] += 1
                self.lemmaStats[lemmas[i]][tags[i]][words[i]] += 1

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

    def backoffLM(self, preLemma, lemma, preLabel, label, tag):
        if (preLemma, lemma) not in self.depStats:
            # Backoff to ungiram + pos model
            if lemma not in self.lemmaStats:
                # return itself
                return lemma
            elif tag not in self.lemmaStats[lemma]:
                # get simple unigram without pos
                return self.getUnigram(self.lemmaStats[lemma])
            else:
                return sorted(self.lemmaStats[lemma][tag].iteritems(),key=lambda (k,v): v,reverse=True)[0][0]
        elif (preLabel, label) not in self.depStats[(preLemma, lemma)]:
            return self.getUnigram(self.lemmaStats[lemma])
        else:
            return sorted(self.depStats[(preLemma, lemma)][(preLabel, label)].iteritems(),\
             key=lambda (k,v): v,reverse=True)[0][0]

def main():
    PARSER = argparse.ArgumentParser(description="Inflect a lemmatized corpus with n-gram model")
    PARSER.add_argument("-t", type=str, default="data/train", help="training data prefix")
    PARSER.add_argument("-l", type=str, default="lemma", help="lemma file suffix")
    PARSER.add_argument("-w", type=str, default="form", help="word file suffix")
    PARSER.add_argument("-d", type=str, default="data/dtest", help="test file")
    PARSER.add_argument("-n", type=int, default=2, help="n-gram model(default bigram)")
    args = PARSER.parse_args()

    # Python sucks at UTF-8
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout) 

    bidep = BiDep(args.t + "." + args.w, args.t + "." + args.l,\
            args.t + ".tag", args.t + ".tree")

    for lemmas, tags, trees in izip(utf8read(args.d + "." + args.l), \
        utf8read(args.d + ".tag"),utf8read(args.d + ".tree")):
        lemmas = lemmas.rstrip().lower().split()
        tags = tags.split()
        tree = DepTree(trees)
        sen = ""
        for i,node in enumerate(tree):
            le = bidep.backoffLM(lemmas[node.parent_index()-1], lemmas[i],\
            tree.node(node.parent_index()).label(), node.label(), tags[i])
            sen += le + " "
        print sen.rstrip()

if __name__ == '__main__':
    main()