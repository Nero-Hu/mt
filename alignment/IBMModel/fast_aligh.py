#!/usr/bin/env python
import optparse
import sys
import numpy
import math
from collections import defaultdict

def main(argv):
	#Shameless copy 
	optparser = optparse.OptionParser()
	optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
	optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
	optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
	optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
	optparser.add_option("-m", "--model", dest="model_type", default=1, type="int", help="Define which model to train ( default=1 )")
	(opts, _) = optparser.parse_args()
	f_data = "%s.%s" % (opts.train, opts.french)
	e_data = "%s.%s" % (opts.train, opts.english)

	sys.stderr.write("Start training...\n")
	bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
	f_set = set()
	e_set = set()
	#Get word counts
	for e,f in bitext:
		e_set.update(e)
		f_set.update(f)

	#Add None word for index
	e_set.add(None)

	f_count = len(f_set)
	e_count = len(e_set)

	#Mapping location in dictionary
	e_index = {}
	f_index = {}
	for (n,f) in enumerate(f_set):
		f_index[f] = n

	for (n,e) in enumerate(e_set):
		e_index[e] = n

	a = []
	tef = trainFastAlign(bitext, f_index, e_index, f_count, e_count, max_iter=5)
	output(bitext, tef, f_index, e_index)

#Train t(e|f) using ibm model 1 for 5 times, to gain better t(e|f)
def trainModel1(bitext, f_index, e_index, f_count, e_count, max_iter = 20, convergeThreshold=1e-2):
	#Construct array as e_count x f_count
	#Probability P(e|f)
	tef = numpy.zeros((e_count,f_count))
	#record new pef to compare with old one
	ntef = numpy.zeros((e_count,f_count))
	#Initialize parameters with uniform distribution
	ntef.fill(float(1)/float(e_count))

	it = 0
	while it < max_iter and sum(sum(numpy.absolute(tef-ntef))) > convergeThreshold:
		it += 1
		tef = ntef
		# Initialize Count for C(e|f)
		cef = numpy.zeros((e_count,f_count))
		totalf = numpy.zeros(f_count)
		for e,f in bitext:
			fn = [None] + f
			#Compute normalization
			for e_word in e:
				totale = float(0)
				for f_word in fn:
					totale += tef[e_index[e_word]][f_index[f_word]]
				for f_word in fn:
					cef[e_index[e_word]][f_index[f_word]] += float(tef[e_index[e_word]][f_index[f_word]]) / float(totale)
					totalf[f_index[f_word]] += float(tef[e_index[e_word]][f_index[f_word]]) / float(totale)
		#Estimate probabilities
		ntef = (cef.T / totalf[:,None]).T

	return ntef

#Take optimized t(e|f), train using IBM Model 2
def trainFastAlign(bitext, f_index, e_index, f_count, e_count, max_iter = 50, convergeThreshold = 1e-2):
	global p0, lamb
	#Initialize, perform 5 times of model 1
	model1_iter = 5
	p0 = 0.08
	lamb = 4.0
	# tef = trainModel1(bitext, f_index, e_index, f_count, e_count, model1_iter)
	tef = numpy.zeros((e_count,f_count))
	tef.fill(1e-9)

	#While not converge
	# while it < max_iter and sum(sum(numpy.absolute(tef-ntef))) > convergeThreshold:
	for it in range(max_iter):
		cef = numpy.zeros((e_count,f_count))
		likelihood = 0.0

		for (n, (e, f)) in enumerate(bitext):
			en = [None] + e
			le = len(en)
			lf = len(f)

			prob_e = numpy.zeros(le)
			for (j, f_word) in enumerate(f):
				sum_prob = 0.0
				#Uniform
				prob_a_i = 1.0 / le
				for (i, e_word) in enumerate(en):
					if i == 0:
						prob_a_i = p0
					else:
						az = computeZ(j+1, lf, le - 1, lamb) / (1.0-p0)
						prob_a_i = computeProb(j+1, i, lf, le - 1, lamb) / az

					prob_e[i] = tef[e_index[e_word]][f_index[f_word]] * prob_a_i
					sum_prob += prob_e[i]

				# print sum_prob
				for (i, e_word) in enumerate(en):
					c = prob_e[i] / sum_prob
					cef[e_index[e_word]][f_index[f_word]] += c
				likelihood += math.log(sum_prob);

		base2_likelihood = likelihood / math.log(2);
		# print base2_likelihood
		#Estimate probabilities
		tef += cef
		totale = numpy.sum(tef, axis = 1)
		# print totale
		for row in range(tef.shape[0]):
			if totale[row] == 0.0:
				totale[row] = 1.0
			tef[row,:] = tef[row,:] / totale[row]
		# print tef


	#return tef and a
	return tef

def computeZ(i, lf, le, lamb):
	split = float(i) * float(le) / float(lf)
	floor = int(split)
	ceil = floor + 1

	ratio = math.exp(-lamb / le)
	num_top = le - floor

	high = 0.0
	low = 0.0
	if num_top != 0:
		high = computeProb(i, ceil, lf, le, lamb) * (1.0 - math.pow(ratio, num_top)) / (1.0 - ratio)
	if floor != 0:
		low = computeProb(i, floor, lf, le, lamb) * (1.0 - math.pow(ratio, floor)) / (1.0 - ratio)

	return high + low

def computeProb(i, j, lf, le, lamb):
	h = -1.0 * abs( float(j) / float(le) - float(i) / float(lf))
	return math.exp(h * lamb)

def output(bitext, tef, f_index, e_index):
	for (n,(e, f)) in enumerate(bitext):
		align = []
		en = [None] + e
		le = len(en)
		lf = len(f)
		for (j, f_word) in enumerate(f):
			max_i = 0
			max_prob = 0
			prob_a_i = 0.0
			for (i, e_word) in enumerate(en):
				if i == 0:
					prob_a_i = p0
				else:
					az = computeZ(j+1, lf, le-1, lamb) / (1.0-p0)
					prob_a_i = computeProb(j+1, i , lf, le-1, lamb) / az

				tProb = tef[e_index[e_word]][f_index[f_word]] * prob_a_i
				if tProb > max_prob:
					max_i = i
					max_prob = tProb
			#if null word write 0
			if max_i > 0:
				sys.stdout.write("%i-%i " % (max_i-1,j))
		sys.stdout.write("\n")

#Main entry
if __name__ == '__main__':
	main(sys.argv)
