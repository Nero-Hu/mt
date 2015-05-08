#!/usr/bin/env python

import codecs
import time
import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
import os.path
from collections import Counter
from collections import OrderedDict
from theano.tensor.signal import downsample

def utf8read(f):
    return codecs.open(f, 'r', 'utf-8')

def utf8write(f):
	return codecs.open(f, 'w', 'utf-8')

class RNN_LM:
	"""
	Recurrent Neural Network

	input_dimension  -   int : dimension of input vector (generally vocabulary size)
	hidden_unit      -   int : hidden unit numbers
	output_dimension -   int : dimension of output vector (generally will be same as input)
	emb_dimesion     -   int : dimension of word embedding (default 100)
	epochs           -   int : max epochs
	"""
	def __init__(self, phrase_pair, input_dimension, hidden_unit, output_dimension, emb_dimesion=500, epochs = 50):
		
	
def main():
	rnn_lm = RNN_LM()

if __name__ == '__main__':
	main()