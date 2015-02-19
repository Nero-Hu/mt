#!/usr/bin/env python
import optparse
import sys
import numpy
from collections import defaultdict
from math import factorial

class HashableDict(dict):
	"""
	This class implements a hashable dict, which can be
	put into a set.
	"""
	def __key(self):
		return tuple((k,self[k]) for k in sorted(self))

	def __hash__(self):
		return hash(self.__key())

	def __eq__(self, other):
		return self.__key() == other.__key()

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

	global f_set
	global e_set

	f_set = set()
	e_set = set()
	#Get word counts
	for e,f in bitext:
		e_set.update(e)
		f_set.update(f)

	#Add None word for index
	f_set.add(None)

	global f_count
	global e_count
	f_count = len(f_set)
	e_count = len(e_set)

	#Mapping location in dictionary
	global e_index
	global f_index
	e_index = {}
	f_index = {}

	for (n,f) in enumerate(f_set):
		f_index[f] = n

	for (n,e) in enumerate(e_set):
		e_index[e] = n

	global tef
	global d
	# tef = trainModel1(bitext, max_iter=5)
	# tef, a = trainModel2(bitext, max_iter=5)
	trainModel3(bitext, max_iter=5)
	output(bitext, tef, d, f_index, e_index)

#Train t(e|f) using ibm model 1 for 5 times, to gain better t(e|f)
def trainModel1(bitext, max_iter = 20, convergeThreshold=1e-2):
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
def trainModel2(bitext, max_iter = 50, convergeThreshold = 1e-2):
	#Initialize, perform 5 times of model 1
	model1_iter = 5
	tef = trainModel1(bitext, model1_iter)
	#Align probabilities for each sentence
	a = []

	it = 0
	#While not converge
	# while it < max_iter and sum(sum(numpy.absolute(tef-ntef))) > convergeThreshold:
	for it in range(max_iter):
		cef = numpy.zeros((e_count,f_count))
		totalf = numpy.zeros(f_count)
		cij = []
		total_a = []
		for (n, (e, f)) in enumerate(bitext):
			le = len(e)
			fn = [None] + f
			lf = len(fn)
			#Initialize a if this is the first time
			if it == 0:
				a.append(numpy.zeros((lf,le)))
				a[n].fill(float(1)/float(lf + 1))
			#Initialize cij
			if len(cij) == n:
				cij.append(numpy.zeros((lf,le)))
				total_a.append(numpy.zeros(le))

			#Compute normalization
			totale = numpy.zeros(le)
			for (j, e_word) in enumerate(e):
				for (i, f_word) in enumerate(fn):
					totale[j] += float(tef[e_index[e_word]][f_index[f_word]]) * float(a[n][i][j])
			#Prevent from deviding by 0
			totale[totale == float(0)] = numpy.inf

			#Collect counts
			for (j, e_word) in enumerate(e):
				for (i, f_word) in enumerate(fn):
					c = float(tef[e_index[e_word]][f_index[f_word]]) * float(a[n][i][j]) / float(totale[j])
					cef[e_index[e_word]][f_index[f_word]] += c
					totalf[f_index[f_word]] += c
					cij[n][i][j] += c
					total_a[n][j] += c
			# #Smooth counts
			# laplace = numpy.min(cij[n][cij[n]!=0])
			# laplace *= 0.5
			# total_a[n] += laplace * le

		#End of collecting
		totalf[totalf == float(0)] = numpy.inf
		#Estimate probabilities
		tef = (cef.T / totalf[:,None]).T
		for n in range(len(total_a)):
			total_a[n][total_a[n] == float(0)] = numpy.inf
			a[n] = (cij[n].T / total_a[n][:,None]).T

	#return tef and a
	return tef, a

#Model3
def trainModel3(bitext, max_iter = 50, convergeThreshold = 1e-2, nullProb = 0.5):

	#Define global distortion probability
	global distortionProb
	global null_prob
	global tef
	global d
	global fertility
	#Initialize, perform 5 times of model 2
	model1_iter = 5
	null_prob = nullProb

	tef, d = trainModel2(bitext, model1_iter)

	fertility = defaultdict(lambda: defaultdict(lambda: float(0.1)))

	for it in range(max_iter):
		max_fert = 0

		distortionProb = []

		cef = numpy.zeros((e_count,f_count))
		total_t = numpy.zeros(f_count)

		count_f = defaultdict(lambda: defaultdict(lambda: 0.0))
		totalf = numpy.zeros(f_count)

		count_d = []
		total_d = []

		count_p0 = float(0)
		count_p1 = float(0)

		for (n, (e, f)) in enumerate(bitext):
			le = len(e)
			fn = [None] + f
			lf = len(fn)

			#Initialize count_d, total_a if this is the first time
			if len(count_d) == n:
				count_d.append(numpy.zeros((le,lf)))
				total_d.append(numpy.zeros(lf))
				distortionProb.append(numpy.zeros((le,lf)))
				distortionProb[n].fill(0.1)

			#Sample alignment
			A = sample(n, e, fn)
			print A

			#Collect counts
			c_total = 0
			for a in A:
				c_total += prob(a, e, fn, null_prob)
			for a in A:
				c = prob(a, e, fn, null_prob) / c_total
				null = 0
				
				for (j, e_word) in enumerate(e):
					#Find corresponding f word using alignment a
					f_word = fn[a[j]]

					#Lexical transition
					cef[e_index[e_word]][f_index[f_word]] += c
					total_t[f_index[f_word]] += c

					#Distortion
					count_d[n][j][a[j]] += c
					total_d[n][a[j]] += c

					#Count null words
					if a[j] == 0:
						null += 1

				#Collect couns for null
				count_p1 += null * c
				count_p0 += (le - 2 * null) * c

				# Collect fertility
				for (i, f_word) in enumerate(fn):
					fert = 0
					for (j, e_word) in enumerate(e):
						if i == a[j]:
							fert += 1

					count_f[fert][f_word] += c
					total_f[f_index[f_word]] += c

					if fert > max_fert:
						max_fert = fert

		#Estimate parameters
		# For translation
		tef = (cef.T / total_t[:,None]).T
		# For distortion
		for n in range(len(total_d)):
			total_d[n][total_d[n] == float(0)] = numpy.inf
			a[n] = (count_d[n].T / total_d[n][:,None]).T
		#For fertility
		for fer in range(max_fert+1):
			for f_word in f_set:
				fertility[fer][f_index[f_word]] = count_f[fer][f_index[f_word]] / total_f[f_index[f_word]]
		# Reestimate for null probability
		p1 = count_p1 / (count_p1 + count_p0)
		null_prob = 1 - p1

#Sample
def sample(n, e, f):
	A = set()

	le = len(e)
	lf = len(f)

	for (j, f_word) in enumerate(f):
		max_j = 0
		max_prob = 0
		for (i, e_word) in enumerate(e):
			a = HashableDict()
			fert = HashableDict()
			# Initialize all fertility to zero
			for x in range(0, lf):
				fert[x] = 0

			#Pegging 1 alignment point
			a[i] = j
			fert[j] = 1

			#Find best alignment
			for (ei, e_word) in enumerate(e):
				if ei != i:
					max_prob = float(0)
					max_j = 1

					for (fj, f_word) in enumerate(f):
						tProb = tef[e_index[e_word]][f_index[f_word]] * d[n][ei][fj]
						if tProb > max_prob:
							max_prob = tProb
							max_j = fj

					a[ei] = max_j
					fert[max_j] += 1
			
			a = hillClimb(n, e, f, a, i, fert)
			neighbor = neighboring(n, e, f, a, i, fert)
			A.update(neighbor)

	return A

#Hill climb
def hillClimb(n, e, f, a, j_p, fert):
	old_a = []
	max_fert = fert

	while old_a != a:
		old_a = a
		for (neighbor, n_fert) in neighboring(n, e, f, a, j_p, fert):
			if prob(n, neighbor, e, f, n_fert) > prob(n, a, e, f, max_fert):
				max_fert = n_fert
				a = neighbor

	return a

#Neighboring
def neighboring(n, e, f, a, j_p, fert):
	N = set()

	le = len(e)
	lf = len(f)

	for (j, e_word) in enumerate(e):
		if j != 0 and j != j_p:
			# Moves
			for (i, f_word) in enumerate(f):
				a_new = HashableDict(a)
				a_new[j] = i

				new_fert = fert
				if new_fert[a[j]] > 0:
					new_fert = HashableDict(fert)
					new_fert[a[j]] -= 1
					new_fert[i] += 1
				#Update set
				N.update([(a_new, new_fert)])

	for (j, e_word) in enumerate(e):
		if j != 0 and j != j_p:
			# Swaps
			for (j_swap, e_word) in enumerate(e):
				if j_swap != j_p and j_swap != j:
					a_new = HashableDict(a)
					a_new[j] = a[j_swap]
					a_new[j_swap] = a[j]

					N.update([(a_new, fert)])

	return N

#Calculate probabilities for alignmeng a
def prob(n, a, e, f, fert):
	le = len(e)
	lf = len(f)

	p1 = 1 - null_prob

	total = float(1)

	#Compute null insertion
	total *= pow(p1, fert[0]) * pow(null_prob, le - 2* fert[0])
	if total ==0:
		return total

	#Compute the combination (le - fert[0]) choose fert[0]
	for i in range(1, fert[0] + 1):
		total *= (le - fert[0] -i + 1) / i
		if total == 0:
			return total

	#Compute fertilities term
	for i in range(1, lf):
		f_word = f[i]
		total *= factorial(fert[i]) * fertility[fert[i]][f_word]
		if total ==0:
			return total

	#Multiply to get probabilities
	for j in range(le):
		e_word = e[j]
		f_word = f[a[j]]

		total *= tef[e_index[e_word]][f_index[f_word]]
		total *= distortionProb[n][j][a[j]]
		if total == 0:
			return total

	return total

def output(bitext, tef, a, f_index, e_index):
	for (n,(e, f)) in enumerate(bitext):
		align = []
		fn = [None] + f
		for (i, e_word) in enumerate(e):
			max_j = 0
			max_prob = 0
			for (j, f_word) in enumerate(fn):
				tProb = tef[e_index[e_word]][f_index[f_word]] * a[n][j][i]
				if tProb > max_prob:
					max_j = j
					max_prob = tProb
			#if null word write 0
			if max_j > 0:
				sys.stdout.write("%i-%i " % (i,max_j-1))
		sys.stdout.write("\n")

#Main entry
if __name__ == '__main__':
	main(sys.argv)
