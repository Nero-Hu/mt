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

class RNN:
	"""
	Recurrent Neural Network

	input_dimension  -   int : dimension of input vector (generally vocabulary size)
	hidden_unit      -   int : hidden unit numbers
	output_dimension -   int : dimension of output vector (generally will be same as input)
	emb_dimesion     -   int : dimension of word embedding (default 100)
	epochs           -   int : max epochs
	"""
	def __init__(self, phrase_pair, input_dimension, hidden_unit, output_dimension, emb_dimesion=500, epochs = 50):
		self.phrase_pair = phrase_pair
		self.input_dimension = input_dimension
		self.hidden_unit = hidden_unit
		self.output_dimension = output_dimension
		self.emb_dimesion = emb_dimesion
		self.epochs = epochs
		#########################
		# Initial parameters sampled from 0 mean gaussian, with 0.01 std
		#########################
		# embedding weights
		self.W_e = theano.shared(np.random.normal(0, 0.01, \
			(input_dimension, emb_dimesion)).astype(theano.config.floatX), name = 'W_e')
		# input weights, decoder and encoder has different weights
		self.W_e_in = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, emb_dimesion)).astype(theano.config.floatX), name = 'W_e_in')
		self.W_d_in = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, emb_dimesion)).astype(theano.config.floatX), name = 'W_d_in')
		# output weights, decoder and encoder has different weights
		self.U_e = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'U_e')
		self.U_d = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'U_d')
		# Summary weights, decoder and encoder has different weights
		self.V_e = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'V_e')
		self.V_d = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'V_d')
		# Summary only on decoder side
		self.C = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'C')
		self.C_z = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'C_z')
		self.C_r = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'C_r')
		# weight for reset gate, decoder and encoder has different weights
		self.W_e_r = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, emb_dimesion)).astype(theano.config.floatX), name = 'W_e_r')
		self.W_d_r = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, emb_dimesion)).astype(theano.config.floatX), name = 'W_d_r')
		self.U_e_r = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'U_e_r')
		self.U_d_r = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'U_d_r')
		# weight for update gate, decoder and encoder has different weights
		self.W_e_z = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, emb_dimesion)).astype(theano.config.floatX), name = 'W_e_z')
		self.W_d_z = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, emb_dimesion)).astype(theano.config.floatX), name = 'W_d_z')
		self.U_e_z = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'U_e_z')
		self.U_d_z = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'U_d_z')
		# weight for output layer
		self.O_h = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'O_h')
		self.O_y = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, output_dimension)).astype(theano.config.floatX), name = 'O_y')
		self.O_c = theano.shared(np.random.normal(0, 0.01, \
			(hidden_unit, hidden_unit)).astype(theano.config.floatX), name = 'O_c')
		# output weight
		self.G = theano.shared(np.random.normal(0, 0.01, \
			(self.input_dimension, int(self.hidden_unit/2))).astype(theano.config.floatX), name = 'G')

		self.params = [self.W_e, self.W_e_in, self.W_d_in, self.U_e, self.U_d, self.V_e, self.V_d, \
			self.C, self.C_z, self.C_r, self.W_e_r, self.W_d_r, self.U_e_r, self.U_d_r, self.W_e_z, self.W_d_z,\
			self.U_e_z, self.U_d_z, self.O_h, self.O_y, self.O_c, self.G]

	def loss(self, source, target):
		"""
		Joint train source phrase and target phrase
		source  -  [vocabulary_size, len(source)] array, where each column represents a word
		target  -  same as above
		"""
		############################
		# First calculate source side
		############################
		# hidden states, init to be 0
		# hs = T.zeros((self.hidden_unit,))
		# Iterating over column of source
		def encoderUpdate(s_word, hs):
			z = T.nnet.sigmoid(T.dot(self.W_e_z, T.dot(self.W_e.T, s_word)) + T.dot(self.U_e_z, hs))
			r = T.nnet.sigmoid(T.dot(self.W_e_r, T.dot(self.W_e.T, s_word)) + T.dot(self.U_e_r, hs))
			hs_t = T.tanh(T.dot(self.W_e_in, T.dot(self.W_e.T, s_word)) + T.dot(self.U_e, r * hs))
			hs = z * hs + (1.-z) * hs_t
			return hs
			# print z.eval(), r.eval()
		hs, _ = theano.scan(fn = encoderUpdate, sequences=source, outputs_info = T.zeros((self.hidden_unit,)))
		self.hs = hs[-1]
		# Update summary 
		self.c = T.tanh(T.dot(self.V_e, self.hs))
		# initial hidden state for hy
		self.hy = T.tanh(T.dot(self.V_d, self.c))
		self.last_y = T.zeros((self.output_dimension,))

		def decoderUpdate(t_word):
			# use summary and weights to back training target phrase
			# Initialize last output 
			# Compute hidden states
			z = T.nnet.sigmoid(T.dot(self.W_d_z, T.dot(self.W_e.T, self.last_y)) + T.dot(self.U_d_z, self.hy) + T.dot(self.C_z, self.c))
			r = T.nnet.sigmoid(T.dot(self.W_d_r, T.dot(self.W_e.T, self.last_y)) + T.dot(self.U_d_r, self.hy) + T.dot(self.C_r, self.c))
			hy_t = T.tanh(T.dot(self.W_d_in, T.dot(self.W_e.T, self.last_y)) + r * (T.dot(self.U_d, self.hy) + T.dot(self.C, self.c)))
			self.hy = z * self.hy + (1.-z) * hy_t
			s_t = T.dot(self.O_h, self.hy) + T.dot(self.O_y, self.last_y) + T.dot(self.O_c, self.c)
			# Max-out pooling
			s = downsample.max_pool_2d(T.reshape(s_t,(int(self.hidden_unit/2),2)), (1, 2))
			# softmax output
			p = T.nnet.softmax(T.flatten(T.dot(self.G, s)))
			# set this word to be last word
			self.last_y = t_word

			return p
		p_given_y, _ = theano.scan(fn = decoderUpdate, sequences=target)

		return p_given_y

	def nll_multiclass(self, p_given_y, target):
		res = p_given_y * target
		return -T.sum(T.log(res[(res > 0).nonzero()]))

	def train(self):
		"""
		f, e is matrix with size [len(f/e), vocabulary]
		"""
		if os.path.isfile("model.gz"):
			self.loadModel()
		else:
			f = T.matrix()
			e = T.matrix()
			p_given_y = self.loss(f, e)
			loss = self.nll_multiclass(p_given_y, e)

			gparams = T.grad(loss, self.params)

			updates = OrderedDict()
			for param, gparam in zip(self.params, gparams):
				upd = -0.01 * gparam
				updates[param] = param + upd

			MSGD = theano.function([f,e], loss, updates=updates)
			print "Start training..."
			epoch = 0
			while epoch < self.epochs:
				########################
				# Update counter
				########################
				epoch += 1
				t0 = time.time()
				totalNLL = 0.
				########################
				# training with sgd
				########################
				for i, (source, target) in enumerate(self.phrase_pair):
					source = source.astype(theano.config.floatX)
					target = target.astype(theano.config.floatX)
					loss = MSGD(source, target)
					totalNLL += loss

				print "Epochs %d, totalNLL: %f, used time: %f" % (epoch, totalNLL, time.time() - t0)

			self.saveModel()

	def rescore(self, source, target):
		"""
		source  -   T.matrix [len(source), vocabulary_size]
		Represent the words to be translated

		return :
		predicted output phrase with its score
		"""
		f = T.matrix()
		e = T.matrix()
		p_given_y = self.loss(f, e)

		# Load could result the float type in-consistancy
		GP = theano.function([f,e], p_given_y, allow_input_downcast=True)

		return GP(source, target)


	def saveModel(self):
		f = open("model.gz", 'wb')
		print 'Dumping model parameters...'
		pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()

	def loadModel(self):
		print 'Found model file, loading model parameters...'
		f = open("model.gz", 'rb')
		self.params = pickle.load(f)
		# for p in self.params:
		# 	print p.name, p.eval()
		f.close() 

def main():
	theano.config.optimizer = "fast_compile"
	max_voc = 1500
	fVoc = Counter()
	eVoc = Counter()
	pair = []
	phrase_pair = []
	# Load data and train with data
	for line in utf8read("tm"):
		(f, e, logprob) = line.rstrip().split(" ||| ")
		for f_word in f.split():
			fVoc[f_word] += 1
		for e_word in e.split():
			eVoc[e_word] += 1
		pair.append((f, e))

	f_dict = {}
	e_dict = {}
	f_dict['unk'] = 0
	e_dict['unk'] = 0
	for k, v in fVoc.most_common(max_voc-1):
		f_dict[k] = len(f_dict)

	for k, v in eVoc.most_common(max_voc-1):
		e_dict[k] = len(e_dict)

	for f, e in pair:
		f, e = f.split(), e.split()
		f_array = np.zeros((len(f),max_voc))
		e_array = np.zeros((len(e),max_voc))
		for i, f_word in enumerate(f):
			if f_word not in f_dict.keys():
				f_array[i][0] = 1
			else:
				f_array[i][f_dict[f_word]] = 1

		for i, e_word in enumerate(e):
			if e_word not in e_dict.keys():
				e_array[i][0] = 1
			else:
				e_array[i][e_dict[e_word]] = 1
		phrase_pair.append((f_array, e_array))

	input_dimension = output_dimension = max_voc
	hidden_unit = 1000
	rnn = RNN(phrase_pair, input_dimension, hidden_unit, output_dimension)
	rnn.train()
	###########################
	# rescore phrase
	###########################
	rescorePhraseFile = utf8write("tm.rescore")
	for (f_phrase, e_phrase), (f, e) in zip(pair, phrase_pair):
		# theano function to get output probability
		prob = rnn.rescore(f,e)
		# score is simply sum of negative log-likelihood
		score = -rnn.nll_multiclass(prob, e)
		rescorePhraseFile.write( "%s ||| %s ||| %f\n" % (f_phrase, e_phrase, score.eval()))


if __name__ == '__main__':
	main()