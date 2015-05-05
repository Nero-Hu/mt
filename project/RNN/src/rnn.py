#!/usr/bin/env python

import time
import numpy as np
import theano
import theano.tensor as T

class RNN:
	"""
	Recurrent Neural Network

	input_dimension  -   int : dimension of input vector (generally vocabulary size)
	hidden_unit      -   int : hidden unit numbers
	output_dimension -   int : dimension of output vector (generally will be same as input)
	emb_dimesion     -   int : dimension of word embedding (default 500)
	epochs           -   int : max epochs
	actFunc          -   
	"""
	def __init__(self, input_dimension, hidden_unit, output_dimension, emb_dimesion=500, epochs = 50, actFunc = T.tanh, errorType = "cross"):
		self.epochs = epochs
		#########################
		# Initial parameters sampled from 0 mean gaussian, with 0.01 std
		#########################
		self.emb = theano.shared(np.random.normal(0, 0.01, \
			(input_dimension+1, emb_dimesion)).astype(theano.config.floatX))
		self.

	def mse(self):
		pass

	def train(self):
		print "Start training..."
		epoch = 0
		while epoch < self.epochs:
			########################
			# Update counter
			########################
			epoch += 1
			t0 = time.time()
			########################
			# training with sgd and adadelta
			########################


			print "Epochs %d, used time: %f" % (epoch, time.time() - t0)

def main():
	input_dimension = 10
	hidden_unit = 1000
	output_dimension = 10
	rnn = RNN(input_dimension, hidden_unit, output_dimension)
	rnn.train()

if __name__ == '__main__':
	main()