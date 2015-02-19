#!/usr/bin/env python
import optparse
import sys
import numpy
import math

def main(argv):
	#Shameless copy 
	optparser = optparse.OptionParser()
	optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
	optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
	optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
	optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
	(opts, _) = optparser.parse_args()
	f_data = "%s.%s" % (opts.train, opts.french)
	e_data = "%s.%s" % (opts.train, opts.english)

	sys.stderr.write("Start training...\n")
	bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
	revtext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(e_data), open(f_data))[:opts.num_sents]]

	e2f = trainFastAlign(bitext, max_iter=5)
	f2e = trainFastAlign(revtext, max_iter=5)

	global A
	A = []

	grow_diagonal(e2f, f2e)
	# finalAnd(e2f)
	# finalAnd(f2e, reverse = True)
	output()

def trainModel1(bitext, f_index, e_index, f_count, e_count, max_iter = 20, convergeThreshold=1e-2):
	"""

	Train model 1
	"""
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
			en = [None] + e
			#Compute normalization
			for e_word in en:
				totale = float(0)
				for f_word in f:
					totale += tef[e_index[e_word]][f_index[f_word]]
				for f_word in f:
					cef[e_index[e_word]][f_index[f_word]] += float(tef[e_index[e_word]][f_index[f_word]]) / float(totale)
					totalf[f_index[f_word]] += float(tef[e_index[e_word]][f_index[f_word]]) / float(totale)
		#Estimate probabilities
		ntef = (cef.T / totalf[:,None]).T

	return ntef

def trainFastAlign(bitext, max_iter = 50, convergeThreshold = 1e-2):
	"""

	Main function of training fast align

	bitext: list
	contains language pair as (target, source)

	max_iter: int
	set max iteration that the model will train

	convergeThreshold: float
	set threshold of convergence, currently un-used.
	"""

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

	global p0, lamb

	#Initialize
	p0 = 0.08
	lamb = 4.0

	# tef using model1 go get better initilization
	tef = trainModel1(bitext, f_index, e_index, f_count, e_count, max_iter = 5)

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

		for row in range(tef.shape[0]):
			if totale[row] == 0.0:
				totale[row] = 1.0
			tef[row,:] = tef[row,:] / totale[row]
		# print tef

	#Align using tef
	return align(bitext, tef, f_index, e_index)

def computeZ(i, lf, le, lamb):
	"""

	Compute normalization constant Z

	i: int
	location of current word

	lf, le: int
	length of target sentence(lf) and source sentence(le)

	lamb: float
	Hyper parameters, currently set to be 4.0

	"""

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
	"""

	Compute numerator

	i,j : int
	alignment of word i to j

	lf, le: int
	length of target sentence(lf) and source sentence(le)

	lamb: float
	Hyper parameters, currently set to be 4.0
	
	"""
	h = -1.0 * abs( float(j) / float(le) - float(i) / float(lf))
	return math.exp(h * lamb)

def align(bitext, tef, f_index, e_index):
	"""

	Compute alignment
	"""
	res = []
	for (n,(e, f)) in enumerate(bitext):
		align = []
		en = [None] + e
		le = len(en)
		lf = len(f)
		#Create array that records the mapping, and initialize all to be False, means unaligned
		res.append(numpy.zeros((le-1,lf), dtype=bool))

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
			#if null word do not write
			if max_i > 0:
				# sys.stdout.write("%i-%i " % (max_i-1,j))
				res[n][max_i-1][j] = True
		# sys.stdout.write("\n")
	return res

def grow_diagonal(e2f, f2e):
	"""
	
	Use grow diagonal to improve, input show be a n length numpy array with shpae (length of e, length of f)
	
	"""
	neighboring = ((-1,0),(0,-1),(1,0),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))

	for n in range(len(e2f)):
		#Find intersection array and union array
		A.append(e2f[n] & f2e[n].T)
		# Below buggy
		U = e2f[n] | f2e[n].T

		oldA = numpy.zeros(A[n].shape, dtype=bool)

		row = A[n].shape[0]
		col = A[n].shape[1]

		while numpy.array_equal(oldA,A[n]) == False:
			#Add new aligns
			oldA = A[n]
			for e in range(row):
				for f in range(col):
					if A[n][e][f] == True:
						for (i,j) in neighboring:
							e_new = e + i
							f_new = f + j
							if 0 <= e_new < row and 0 <= f_new < col:
								if (numpy.sum(A[n][e_new,:]) == 0 or numpy.sum(A[n][:,f_new]) == 0) and U[e_new][f_new] == True:
									A[n][e_new][f_new] = True

def finalAnd(e2f_in, reverse = False):
	for n in range(len(e2f_in)):
		e2f = e2f_in[n]
		if reverse == True:
			e2f = e2f_in[n].T

		row = A[n].shape[0]
		col = A[n].shape[1]

		for e_new in range(row):
			for f_new in range(col):
				if (numpy.sum(A[n][e_new,:]) == 0 or numpy.sum(A[n][:,f_new]) == 0) and e2f[e_new][f_new] == True:
					A[n][e_new][f_new] = True

def output():
	"""

	Output result in source-target pairs
	"""
	for n in range(len(A)):
		row = A[n].shape[0]
		col = A[n].shape[1]

		for i in range(row):
			for j in range(col):
				if A[n][i][j] == True:
					sys.stdout.write("%i-%i " % (i, j))
					break
		sys.stdout.write("\n")

#Main entry
if __name__ == '__main__':
	main(sys.argv)
